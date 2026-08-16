# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Core plus cross-frame coverage/strength Stable masks with raw-score budget fill."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from cosmos_framework.scripts.robolab_v5_2_motion_core_stable_adaptive_velocity_cache import (
    _select_mask,
    build_v52_ablation_plan,
)
from cosmos_framework.scripts.robolab_v7_direct_core96_weighted_g1_b0 import (
    ACTION_HORIZON_WEIGHTS,
    CORE_TOKEN_BUDGET,
    PROFILE_BLOCK_GROUPS,
    STABLE_BUDGETS,
    TOKEN_BUDGETS,
    V7DirectCore96WeightedG1B0Controller,
)

STRATEGY_VERSION = "v8-core96-coverage-stable-fill"
STABLE_TOP_R = 96
STABLE_MIN_FRAME_COUNT = 6
STABLE_STRENGTH_ORDER = 3


def coverage_strength_stable_masks(
    *,
    torch: Any,
    group_raw_scores: Any,
    stable_budgets: Sequence[int],
    top_r: int,
    min_frame_count: int,
    strength_order: int,
) -> dict[str, Any]:
    """Select nested Stable masks by cross-frame Top-R coverage and robust raw strength."""

    if group_raw_scores.ndim != 3 or int(group_raw_scores.shape[1]) != 8:
        raise ValueError("Expected group raw scores [group,8,spatial]")
    groups, frames, spatial = map(int, group_raw_scores.shape)
    budgets = tuple(int(value) for value in stable_budgets)
    if len(budgets) != groups or any(value < 0 for value in budgets):
        raise ValueError("Stable budget count must match groups and be non-negative")
    if any(budgets[index] < budgets[index + 1] for index in range(groups - 1)):
        raise ValueError("Stable budgets must be non-increasing")
    if not 0 < int(top_r) <= spatial:
        raise ValueError("Stable Top-R must fit the spatial token count")
    if not 1 <= int(min_frame_count) <= frames:
        raise ValueError("Stable minimum frame count must fit the future-frame count")
    if not 1 <= int(strength_order) <= frames:
        raise ValueError("Stable strength order must fit the future-frame count")

    top_r_masks = torch.zeros_like(group_raw_scores, dtype=torch.bool)
    for group in range(groups):
        for frame in range(frames):
            top_r_masks[group, frame] = _select_mask(
                torch,
                group_raw_scores[group, frame],
                int(top_r),
            )
    coverage_counts = top_r_masks.sum(dim=1)
    coverage = coverage_counts.to(group_raw_scores.dtype) / float(frames)
    candidates = coverage_counts >= int(min_frame_count)
    strength = group_raw_scores.sort(dim=1).values[:, int(strength_order) - 1]

    stable_masks = torch.zeros_like(strength, dtype=torch.bool)
    candidate_shortfalls: list[int] = [0] * groups
    for group in reversed(range(groups)):
        inherited = (
            torch.zeros(spatial, dtype=torch.bool, device=group_raw_scores.device)
            if group + 1 == groups
            else stable_masks[group + 1].clone()
        )
        need = budgets[group] - int(inherited.sum())
        if need < 0:
            raise RuntimeError("Inherited Stable mask exceeds the outer group budget")
        allowed = candidates[group] & ~inherited
        take = min(need, int(allowed.sum()))
        selected = _select_mask(torch, strength[group], take, allowed)
        stable_masks[group] = inherited | selected
        candidate_shortfalls[group] = need - take

    return {
        "top_r_masks": top_r_masks,
        "coverage_counts": coverage_counts,
        "coverage": coverage,
        "candidate_masks": candidates,
        "strength": strength,
        "stable_masks": stable_masks,
        "candidate_shortfalls": candidate_shortfalls,
    }


def compose_core_stable_fill_masks(
    *,
    torch: Any,
    core_masks: Any,
    stable_masks: Any,
    group_raw_scores: Any,
    token_budgets: Sequence[int],
) -> dict[str, Any]:
    """Union Core/Stable, inherit deeper masks, then fill exact budgets with raw scores."""

    groups, frames, spatial = map(int, group_raw_scores.shape)
    budgets = tuple(int(value) for value in token_budgets)
    if tuple(core_masks.shape) != (frames, spatial) or tuple(stable_masks.shape) != (groups, spatial):
        raise ValueError("Core/Stable geometry does not match group raw scores")
    if len(budgets) != groups or any(budgets[index] < budgets[index + 1] for index in range(groups - 1)):
        raise ValueError("Token budgets must match groups and be non-increasing")

    masks = torch.zeros_like(group_raw_scores, dtype=torch.bool)
    labels = torch.zeros_like(group_raw_scores, dtype=torch.int8)
    base_counts = torch.zeros((groups, frames), dtype=torch.long, device=group_raw_scores.device)
    overlap_counts = torch.zeros_like(base_counts)
    fill_counts = torch.zeros_like(base_counts)
    for group in reversed(range(groups)):
        for frame in range(frames):
            core = core_masks[frame]
            stable = stable_masks[group]
            inherited = (
                torch.zeros(spatial, dtype=torch.bool, device=group_raw_scores.device)
                if group + 1 == groups
                else masks[group + 1, frame]
            )
            base = core | stable | inherited
            base_count = int(base.sum())
            need = budgets[group] - base_count
            if need < 0:
                raise RuntimeError(f"Core/Stable/inherited tokens exceed G{group + 1} budget")
            fill = _select_mask(torch, group_raw_scores[group, frame], need, ~base)
            masks[group, frame] = base | fill
            labels[group, frame][fill] = 4
            labels[group, frame][stable] = 2
            labels[group, frame][core] = 1
            base_counts[group, frame] = base_count
            overlap_counts[group, frame] = int((core & stable).sum())
            fill_counts[group, frame] = need

    expected = torch.tensor(budgets, device=masks.device)[:, None].expand(groups, frames)
    if not torch.equal(masks.sum(dim=-1), expected):
        raise RuntimeError("Core/Stable/Fill masks did not meet exact budgets")
    if bool((masks[2] & ~masks[1]).any()) or bool((masks[1] & ~masks[0]).any()):
        raise RuntimeError("Core/Stable/Fill masks are not nested")
    return {
        "masks": masks,
        "labels": labels,
        "base_counts": base_counts,
        "core_stable_overlap_counts": overlap_counts,
        "fill_counts": fill_counts,
    }


def build_core_stable_coverage_fill_plan(
    *,
    torch: Any,
    profile_records: Sequence[Mapping[str, Any]],
    block_groups: Sequence[tuple[int, int]] = PROFILE_BLOCK_GROUPS,
    token_budgets: Sequence[int] = TOKEN_BUDGETS,
    stable_budgets: Sequence[int] = STABLE_BUDGETS,
    core_block_range: tuple[int, int] = (4, 27),
    core_block_count: int = 6,
    core_token_budget: int = CORE_TOKEN_BUDGET,
    top_r: int = STABLE_TOP_R,
    min_frame_count: int = STABLE_MIN_FRAME_COUNT,
    strength_order: int = STABLE_STRENGTH_ORDER,
) -> dict[str, Any]:
    """Reuse the validated Core/profile statistics, replacing Adaptive with Stable+Fill."""

    groups = len(tuple(block_groups))
    plan = build_v52_ablation_plan(
        torch=torch,
        profile_records=profile_records,
        block_groups=block_groups,
        token_budgets=token_budgets,
        stable_budgets=(0,) * groups,
        core_block_range=core_block_range,
        core_block_count=core_block_count,
        core_token_budget=core_token_budget,
        stable_reference_core_token_budget=core_token_budget,
    )
    stable = coverage_strength_stable_masks(
        torch=torch,
        group_raw_scores=plan["group_raw_scores"],
        stable_budgets=stable_budgets,
        top_r=top_r,
        min_frame_count=min_frame_count,
        strength_order=strength_order,
    )
    composed = compose_core_stable_fill_masks(
        torch=torch,
        core_masks=plan["core_masks"],
        stable_masks=stable["stable_masks"],
        group_raw_scores=plan["group_raw_scores"],
        token_budgets=token_budgets,
    )

    raw = plan["raw_profiles"]
    branches = plan["branches"]
    blocks = plan["blocks"]
    eps = torch.finfo(raw.dtype).eps
    block_to_group = {
        block: group for group, (start, end) in enumerate(block_groups) for block in range(start, end + 1)
    }
    retention_rows = []
    for branch_index, branch in enumerate(branches):
        for block_index, block in enumerate(blocks):
            group = block_to_group[block]
            for frame in range(8):
                values = raw[branch_index, block_index, frame]
                mass = float(values.sum())
                retained = float(values[composed["masks"][group, frame]].sum())
                retention_rows.append(
                    {
                        "mode": "c",
                        "branch": branch,
                        "block": block,
                        "group": group,
                        "frame": frame + 1,
                        "raw_future_frame_mass": mass,
                        "retained_mass": retained,
                        "attention_mass_retention": retained / max(mass, float(eps)),
                    }
                )
    replacement_rows = [
        {
            "group": group,
            "frame": frame + 1,
            "forced_replacements": 0,
            "threshold_proposed": 0,
            "threshold_accepted": 0,
            "threshold_rejected": 0,
        }
        for group in range(groups)
        for frame in range(8)
    ]
    plan.update(
        {
            "stable_scores": stable["strength"],
            "stable_masks": stable["stable_masks"],
            "stable_top_r_masks": stable["top_r_masks"],
            "stable_coverage_counts": stable["coverage_counts"],
            "stable_coverage": stable["coverage"],
            "stable_candidate_masks": stable["candidate_masks"],
            "stable_strength": stable["strength"],
            "stable_candidate_shortfalls": stable["candidate_shortfalls"],
            "adaptive_scores": torch.zeros_like(plan["group_raw_scores"]),
            "execution_masks": {"c": composed["masks"]},
            "category_labels": {"c": composed["labels"]},
            "core_stable_base_counts": composed["base_counts"],
            "core_stable_overlap_counts": composed["core_stable_overlap_counts"],
            "fill_counts": composed["fill_counts"],
            "replacement_rows": replacement_rows,
            "retention_rows": retention_rows,
        }
    )
    return plan


class V8CoreStableCoverageFillController(V7DirectCore96WeightedG1B0Controller):
    """Direct Core-96 plus coverage/strength Stable and exact raw-score Fill."""

    def _finalize_roi_if_ready(self) -> None:
        if self.execution_masks is not None or self._current is None:
            return
        if int(self._current["step"]) != 0 or str(self._current["branch"]) != "conditional":
            return
        self._materialize_profiles_once()
        expected = sum(end - start + 1 for start, end in self.profile_block_groups)
        if len(self.profile_records) != expected:
            raise RuntimeError(f"Expected {expected} conditional profiles, got {len(self.profile_records)}")
        self.v52_plan = build_core_stable_coverage_fill_plan(
            torch=self.torch,
            profile_records=self.profile_records,
            block_groups=self.profile_block_groups,
            token_budgets=self.token_budgets,
            stable_budgets=self.stable_budgets,
            core_block_range=self.core_block_range,
            core_block_count=self.core_block_count,
            core_token_budget=self.core_token_budget,
        )
        self.normalized_scores = self.v52_plan["v51_normalized"]
        self.smoothed_scores = self.v52_plan["v51_smoothed"]
        self.depth_scores = self.v52_plan["v51_depth_scores"]
        self.execution_masks = self.v52_plan["execution_masks"]["c"]

    def finish(self) -> dict[str, Any]:
        summary = super().finish()
        if self.v52_plan is None:
            raise RuntimeError("V8 plan is unavailable")
        overlap = self.v52_plan["core_stable_overlap_counts"]
        fill = self.v52_plan["fill_counts"]
        coverage_counts = self.v52_plan["stable_coverage_counts"]
        stable_counts = self.v52_plan["stable_masks"].sum(dim=-1)
        summary.update(
            {
                "strategy_version": STRATEGY_VERSION,
                "experiment": "core96_coverage_strength_stable_raw_fill",
                "adaptive_enabled": False,
                "budget_fill_enabled": True,
                "stable_top_r": STABLE_TOP_R,
                "stable_min_frame_count": STABLE_MIN_FRAME_COUNT,
                "stable_coverage_threshold": STABLE_MIN_FRAME_COUNT / 8.0,
                "stable_strength_order": STABLE_STRENGTH_ORDER,
                "stable_selection": "third-smallest raw score among positions in per-frame Top-R for at least 6/8 frames",
                "core_stable_overlap_allowed": True,
                "stable_nested_across_groups": True,
                "stable_counts": stable_counts.tolist(),
                "stable_candidate_counts": (coverage_counts >= STABLE_MIN_FRAME_COUNT).sum(dim=-1).tolist(),
                "stable_candidate_shortfalls": self.v52_plan["stable_candidate_shortfalls"],
                "core_stable_overlap_counts": overlap.tolist(),
                "fill_counts": fill.tolist(),
            }
        )
        return summary
