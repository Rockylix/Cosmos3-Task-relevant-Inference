# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""V5.2 motion-core and temporally stable environment ROI experiment.

Step 0 remains dense.  Raw Action-Q/Future-K attention probabilities are used
to build four paired ablations (A-D) with identical K80 budgets.  Every arm is
constructed from G3 towards G1, making G3 a strict subset of G2 and G2 a
strict subset of G1.  The sparse execution and guided step-0 velocity cache are
inherited unchanged from V5.1.
"""

from __future__ import annotations

import csv
import json
import math
from typing import Any, Mapping, Sequence

from cosmos_framework.scripts.robolab_grouped_temporal_closed_roi_velocity_cache import (
    DEFAULT_BLOCK_GROUPS,
    DEFAULT_DEPTH_LOOKAHEAD_DECAY,
    DEFAULT_TEMPORAL_SCORE_WEIGHTS,
    GroupedTemporalClosedROISparseController,
    GroupedTemporalClosedVelocityCacheSampler,
    build_grouped_score_smoothed_budgeted_masks,
)

VELOCITY_CACHE_STRATEGY_VERSION = "v5.2"
DEFAULT_K80_BUDGETS = (192, 160, 144)
DEFAULT_STABLE_BUDGETS = (88, 72, 64)
DEFAULT_CORE_BLOCK_RANGE = (4, 23)
DEFAULT_CORE_BLOCK_COUNT = 6
DEFAULT_CORE_TOKEN_BUDGET = 48
DEFAULT_STABLE_CV_PENALTY = 1.0
DEFAULT_REPLACEMENT_RELATIVE_THRESHOLD = 0.05
DEFAULT_MAX_REPLACEMENTS = 8
ABLATION_MODES = ("a", "b", "c", "d")


def action_aligned_future_raw_profiles(
    *,
    torch: Any,
    q_gen: Any,
    k_ar: Any,
    k_gen: Any,
    scaling: float,
    token_layout: Mapping[str, Any],
    validate: bool = True,
) -> Any:
    """Return raw Action-aligned future attention, shape ``[8, spatial]``.

    Unlike the V5.1 ranking profile, the spatial dimension is not normalized.
    Consequently each frame sum retains its true attention probability mass.
    """

    if q_gen.ndim != 3 or k_ar.ndim != 3 or k_gen.ndim != 3:
        raise ValueError("Expected Q/K [tokens,heads,head_dim]")
    if int(q_gen.shape[0]) != int(token_layout["num_gen_tokens"]):
        raise RuntimeError("V5.2 Step-0 profile requires a complete GEN sequence")
    q_heads = int(q_gen.shape[1])
    kv_heads = int(k_gen.shape[1])
    if q_heads % kv_heads:
        raise RuntimeError(f"Invalid GQA geometry Hq={q_heads}, Hkv={kv_heads}")

    predicted = [query for query in token_layout["action_queries"] if query["query_role"] == "predicted"]
    horizon_to_position = {int(query["action_horizon"]): int(query["gen_position"]) for query in predicted}
    if sorted(horizon_to_position) != list(range(32)):
        raise RuntimeError("Expected predicted Action horizons 0..31")
    action_index = torch.tensor(
        [horizon_to_position[horizon] for horizon in range(32)], dtype=torch.long, device=q_gen.device
    )
    q = q_gen.index_select(0, action_index).detach().float().permute(1, 0, 2).contiguous()
    k = torch.cat((k_ar, k_gen), dim=0).detach().float()
    k = k.repeat_interleave(q_heads // kv_heads, dim=1).permute(1, 0, 2).contiguous()
    probabilities = torch.softmax(torch.matmul(q, k.transpose(1, 2)) * float(scaling), dim=-1)
    head_mean = probabilities.mean(dim=0)
    if validate and not bool(torch.isfinite(head_mean).all()):
        raise RuntimeError("V5.2 Action attention profile contains NaN/Inf")

    num_ar = int(k_ar.shape[0])
    profiles = []
    for latent in range(1, 9):
        horizons = list(range(4 * (latent - 1), 4 * latent))
        positions = torch.tensor(token_layout["latent_positions"][f"L{latent}"], dtype=torch.long, device=q_gen.device)
        profiles.append(head_mean[horizons].index_select(-1, positions + num_ar).mean(dim=0))
    result = torch.stack(profiles)
    if validate and (not bool(torch.isfinite(result).all()) or bool((result < 0).any())):
        raise RuntimeError("V5.2 raw attention profile is invalid")
    return result


def _select_mask(torch: Any, scores: Any, count: int, allowed: Any | None = None) -> Any:
    if scores.ndim != 1:
        raise ValueError(f"Expected one-dimensional scores, got {tuple(scores.shape)}")
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if count == 0:
        return mask
    candidates = (
        torch.arange(scores.numel(), device=scores.device)
        if allowed is None
        else torch.nonzero(allowed, as_tuple=False).flatten()
    )
    if count < 0 or count > int(candidates.numel()):
        raise ValueError(f"Cannot select {count} from {int(candidates.numel())} candidates")
    chosen = torch.topk(scores.index_select(0, candidates), count, sorted=False).indices
    mask[candidates.index_select(0, chosen)] = True
    return mask


def _records_tensor(
    *, torch: Any, profile_records: Sequence[Mapping[str, Any]], block_groups: Sequence[tuple[int, int]]
) -> tuple[Any, list[str], list[int]]:
    if not profile_records:
        raise ValueError("At least one raw Step-0 profile is required")
    branches = sorted({str(record["branch"]) for record in profile_records})
    blocks = list(range(block_groups[0][0], block_groups[-1][1] + 1))
    first = profile_records[0]["profiles"]
    if first.ndim != 2 or int(first.shape[0]) != 8:
        raise ValueError(f"Expected raw profiles [8,spatial], got {tuple(first.shape)}")
    spatial = int(first.shape[1])
    result = torch.empty((len(branches), len(blocks), 8, spatial), dtype=first.dtype, device=first.device)
    lookup = {(str(record["branch"]), int(record["block"])): record["profiles"] for record in profile_records}
    if len(lookup) != len(profile_records):
        raise RuntimeError("Duplicate V5.2 branch/block profile")
    for branch_index, branch in enumerate(branches):
        for block_index, block in enumerate(blocks):
            value = lookup.get((branch, block))
            if value is None:
                raise RuntimeError(f"Missing V5.2 profile branch={branch} block={block}")
            if tuple(value.shape) != (8, spatial):
                raise RuntimeError("V5.2 profile geometry changed")
            result[branch_index, block_index] = value
    if not bool(torch.isfinite(result).all()) or bool((result < 0).any()):
        raise RuntimeError("V5.2 profile tensor contains NaN/Inf or negative values")
    return result, branches, blocks


def _outward_ranked_masks(
    *, torch: Any, scores: Any, budgets: Sequence[int], core_masks: Any | None = None
) -> tuple[Any, Any]:
    """Construct exact masks from G3 to G1 and label selected token origin."""

    groups, frames, spatial = map(int, scores.shape)
    masks = torch.zeros_like(scores, dtype=torch.bool)
    labels = torch.zeros_like(scores, dtype=torch.int8)
    for group in reversed(range(groups)):
        for frame in range(frames):
            base = torch.zeros(spatial, dtype=torch.bool, device=scores.device)
            if group + 1 < groups:
                base |= masks[group + 1, frame]
                labels[group, frame][base] = labels[group + 1, frame][base]
            if core_masks is not None:
                base |= core_masks[frame]
                labels[group, frame][core_masks[frame]] = 1
            need = int(budgets[group]) - int(base.sum())
            if need < 0:
                raise RuntimeError(f"Mandatory V5.2 tokens exceed G{group + 1} budget")
            selected = _select_mask(torch, scores[group, frame], need, ~base)
            masks[group, frame] = base | selected
            labels[group, frame][selected] = 4
    return masks, labels


def _stable_masks(*, torch: Any, stable_scores: Any, stable_budgets: Sequence[int], forbidden: Any) -> Any:
    groups, spatial = map(int, stable_scores.shape)
    result = torch.zeros_like(stable_scores, dtype=torch.bool)
    for group in reversed(range(groups)):
        base = torch.zeros(spatial, dtype=torch.bool, device=stable_scores.device)
        if group + 1 < groups:
            base |= result[group + 1]
        need = int(stable_budgets[group]) - int(base.sum())
        allowed = ~(base | forbidden)
        result[group] = base | _select_mask(torch, stable_scores[group], need, allowed)
    return result


def _outward_component_masks(
    *,
    torch: Any,
    raw_group_scores: Any,
    adaptive_scores: Any,
    budgets: Sequence[int],
    core_masks: Any,
    stable_masks: Any,
) -> tuple[Any, Any]:
    """Build C masks using positive Adaptive scores then raw-score fill."""

    groups, frames, spatial = map(int, raw_group_scores.shape)
    masks = torch.zeros_like(raw_group_scores, dtype=torch.bool)
    labels = torch.zeros_like(raw_group_scores, dtype=torch.int8)
    for group in reversed(range(groups)):
        for frame in range(frames):
            base = core_masks[frame] | stable_masks[group]
            labels[group, frame][core_masks[frame]] = 1
            labels[group, frame][stable_masks[group]] = 2
            if group + 1 < groups:
                inherited = masks[group + 1, frame]
                base |= inherited
                unset = labels[group, frame] == 0
                labels[group, frame][inherited & unset] = labels[group + 1, frame][inherited & unset]
            need = int(budgets[group]) - int(base.sum())
            if need < 0:
                raise RuntimeError(f"Core/stable/inherited tokens exceed G{group + 1} budget")
            positive = (adaptive_scores[group, frame] > 0) & ~base
            adaptive_count = min(need, int(positive.sum()))
            adaptive = _select_mask(torch, adaptive_scores[group, frame], adaptive_count, positive)
            need -= adaptive_count
            fill = _select_mask(torch, raw_group_scores[group, frame], need, ~(base | adaptive))
            masks[group, frame] = base | adaptive | fill
            labels[group, frame][adaptive] = 3
            labels[group, frame][fill] = 4
    return masks, labels


def _threshold_stabilize_masks(
    *,
    torch: Any,
    c_masks: Any,
    c_labels: Any,
    raw_group_scores: Any,
    budgets: Sequence[int],
    core_masks: Any,
    stable_masks: Any,
    relative_threshold: float,
    max_replacements: int,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    """Stabilize C over L1->L8 while preserving mandatory and nested tokens."""

    groups, frames, spatial = map(int, c_masks.shape)
    result = torch.zeros_like(c_masks)
    labels = torch.zeros_like(c_labels)
    rows: list[dict[str, Any]] = []
    for group in reversed(range(groups)):
        for frame in range(frames):
            mandatory = core_masks[frame] | stable_masks[group]
            labels[group, frame][core_masks[frame]] = 1
            labels[group, frame][stable_masks[group]] = 2
            if group + 1 < groups:
                inherited = result[group + 1, frame]
                mandatory |= inherited
                unset = labels[group, frame] == 0
                labels[group, frame][inherited & unset] = labels[group + 1, frame][inherited & unset]
            capacity = int(budgets[group]) - int(mandatory.sum())
            if capacity < 0:
                raise RuntimeError(f"Mandatory D tokens exceed G{group + 1} budget")
            forced = 0
            accepted = 0
            proposed = 0
            if frame == 0:
                optional = c_masks[group, frame] & ~mandatory
                if int(optional.sum()) > capacity:
                    optional = _select_mask(torch, raw_group_scores[group, frame], capacity, optional)
                missing = capacity - int(optional.sum())
                optional |= _select_mask(torch, raw_group_scores[group, frame], missing, ~(mandatory | optional))
            else:
                previous = result[group, frame - 1]
                entering_mandatory = mandatory & ~previous
                forced = int(entering_mandatory.sum())
                carry = previous & ~mandatory
                if int(carry.sum()) > capacity:
                    carry = _select_mask(torch, raw_group_scores[group, frame], capacity, carry)
                optional = carry.clone()
                missing = capacity - int(optional.sum())
                desired = c_masks[group, frame] & ~mandatory & ~optional
                if missing:
                    chosen = _select_mask(
                        torch,
                        raw_group_scores[group, frame],
                        min(missing, int(desired.sum())),
                        desired,
                    )
                    optional |= chosen
                    missing -= int(chosen.sum())
                if missing:
                    optional |= _select_mask(torch, raw_group_scores[group, frame], missing, ~(mandatory | optional))

                candidates = c_masks[group, frame] & ~mandatory & ~optional
                proposed = int(candidates.sum())
                for _ in range(max_replacements):
                    if not bool(candidates.any()) or not bool(optional.any()):
                        break
                    new_mask = _select_mask(torch, raw_group_scores[group, frame], 1, candidates)
                    old_mask = _select_mask(torch, -raw_group_scores[group, frame], 1, optional)
                    new_index = int(torch.nonzero(new_mask, as_tuple=False)[0])
                    old_index = int(torch.nonzero(old_mask, as_tuple=False)[0])
                    new_score = float(raw_group_scores[group, frame, new_index])
                    old_score = float(raw_group_scores[group, frame, old_index])
                    if new_score <= (1.0 + relative_threshold) * old_score:
                        break
                    optional[old_index] = False
                    optional[new_index] = True
                    candidates[new_index] = False
                    accepted += 1
            result[group, frame] = mandatory | optional
            source_labels = c_labels[group, frame]
            optional_adaptive = optional & (source_labels == 3)
            labels[group, frame][optional_adaptive] = 3
            labels[group, frame][optional & (labels[group, frame] == 0)] = 4
            rows.append(
                {
                    "group": group,
                    "frame": frame + 1,
                    "forced_replacements": forced,
                    "threshold_proposed": proposed,
                    "threshold_accepted": accepted,
                    "threshold_rejected": max(0, proposed - accepted),
                }
            )
    return result, labels, rows


def build_v52_ablation_plan(
    *,
    torch: Any,
    profile_records: Sequence[Mapping[str, Any]],
    block_groups: Sequence[tuple[int, int]] = DEFAULT_BLOCK_GROUPS,
    token_budgets: Sequence[int] = DEFAULT_K80_BUDGETS,
    stable_budgets: Sequence[int] = DEFAULT_STABLE_BUDGETS,
    core_block_range: tuple[int, int] = DEFAULT_CORE_BLOCK_RANGE,
    core_block_count: int = DEFAULT_CORE_BLOCK_COUNT,
    core_token_budget: int = DEFAULT_CORE_TOKEN_BUDGET,
    stable_cv_penalty: float = DEFAULT_STABLE_CV_PENALTY,
    replacement_relative_threshold: float = DEFAULT_REPLACEMENT_RELATIVE_THRESHOLD,
    max_replacements: int = DEFAULT_MAX_REPLACEMENTS,
    temporal_weights: tuple[float, float, float] = DEFAULT_TEMPORAL_SCORE_WEIGHTS,
    depth_lookahead_decay: float = DEFAULT_DEPTH_LOOKAHEAD_DECAY,
) -> dict[str, Any]:
    raw, branches, blocks = _records_tensor(torch=torch, profile_records=profile_records, block_groups=block_groups)
    groups, frames, spatial = len(block_groups), 8, int(raw.shape[-1])
    budgets = tuple(int(value) for value in token_budgets)
    stable_counts = tuple(int(value) for value in stable_budgets)
    if len(budgets) != groups or len(stable_counts) != groups:
        raise ValueError("V5.2 budget count must match block groups")
    if any(budgets[index] < budgets[index + 1] for index in range(groups - 1)):
        raise ValueError("V5.2 token budgets must be non-increasing")
    if any(stable_counts[index] < stable_counts[index + 1] for index in range(groups - 1)):
        raise ValueError("V5.2 stable budgets must be non-increasing")
    if not 0 < core_token_budget <= budgets[-1]:
        raise ValueError("Core budget must fit the smallest group")
    if core_block_count <= 0 or max_replacements < 0 or replacement_relative_threshold < 0:
        raise ValueError("Invalid V5.2 core/replacement parameters")

    eps = torch.finfo(raw.dtype).eps
    frame_mass = raw.sum(dim=-1)
    normalized = raw / frame_mass.unsqueeze(-1).clamp_min(eps)
    entropy_maps = -(normalized * normalized.clamp_min(eps).log()).sum(dim=-1) / math.log(spatial)
    entropy_maps = torch.where(frame_mass > eps, entropy_maps, torch.ones_like(entropy_maps))
    block_mass = frame_mass.mean(dim=(0, 2))
    block_entropy = entropy_maps.mean(dim=(0, 2))
    block_quality = block_mass * (1.0 - block_entropy)

    eligible = [index for index, block in enumerate(blocks) if core_block_range[0] <= block <= core_block_range[1]]
    if core_block_count > len(eligible):
        raise ValueError("Core block count exceeds eligible middle blocks")
    eligible_tensor = torch.tensor(eligible, dtype=torch.long, device=raw.device)
    chosen_local = torch.topk(block_quality.index_select(0, eligible_tensor), core_block_count).indices
    core_block_indices = eligible_tensor.index_select(0, chosen_local)
    core_quality = block_quality.index_select(0, core_block_indices)
    core_weights = core_quality / core_quality.sum().clamp_min(eps)
    branch_max = raw.amax(dim=0)
    core_scores = (branch_max.index_select(0, core_block_indices) * core_weights[:, None, None]).sum(dim=0)
    core_masks = torch.stack([_select_mask(torch, core_scores[frame], core_token_budget) for frame in range(frames)])

    group_raw_scores = torch.empty((groups, frames, spatial), dtype=raw.dtype, device=raw.device)
    v51_group_scores = torch.empty_like(group_raw_scores)
    for group, (start, end) in enumerate(block_groups):
        indexes = torch.tensor(
            [index for index, block in enumerate(blocks) if start <= block <= end],
            dtype=torch.long,
            device=raw.device,
        )
        quality = block_quality.index_select(0, indexes)
        if float(quality.sum()) <= float(eps):
            quality = torch.ones_like(quality)
        weights = quality / quality.sum()
        group_raw_scores[group] = (branch_max.index_select(0, indexes) * weights[:, None, None]).sum(dim=0)
        v51_group_scores[group] = normalized.index_select(1, indexes).amax(dim=(0, 1))

    v51_normalized, v51_smoothed, v51_depth_scores, _ = build_grouped_score_smoothed_budgeted_masks(
        torch=torch,
        scores=v51_group_scores,
        token_budgets=budgets,
        temporal_weights=temporal_weights,
        depth_lookahead_decay=depth_lookahead_decay,
    )
    a_masks, a_labels = _outward_ranked_masks(torch=torch, scores=v51_depth_scores, budgets=budgets)
    b_masks, b_labels = _outward_ranked_masks(
        torch=torch, scores=v51_depth_scores, budgets=budgets, core_masks=core_masks
    )

    mean = group_raw_scores.mean(dim=1)
    std = group_raw_scores.std(dim=1, unbiased=False)
    cv = std / mean.clamp_min(eps)
    stable_scores = mean / (1.0 + float(stable_cv_penalty) * cv)
    core_union = core_masks.any(dim=0)
    stable_masks = _stable_masks(
        torch=torch, stable_scores=stable_scores, stable_budgets=stable_counts, forbidden=core_union
    )
    adaptive_scores = (group_raw_scores - mean[:, None, :]).clamp_min(0)
    c_masks, c_labels = _outward_component_masks(
        torch=torch,
        raw_group_scores=group_raw_scores,
        adaptive_scores=adaptive_scores,
        budgets=budgets,
        core_masks=core_masks,
        stable_masks=stable_masks,
    )
    d_masks, d_labels, replacement_rows = _threshold_stabilize_masks(
        torch=torch,
        c_masks=c_masks,
        c_labels=c_labels,
        raw_group_scores=group_raw_scores,
        budgets=budgets,
        core_masks=core_masks,
        stable_masks=stable_masks,
        relative_threshold=float(replacement_relative_threshold),
        max_replacements=int(max_replacements),
    )

    masks = {"a": a_masks, "b": b_masks, "c": c_masks, "d": d_masks}
    labels = {"a": a_labels, "b": b_labels, "c": c_labels, "d": d_labels}
    for mode, value in masks.items():
        counts = value.sum(dim=-1)
        expected = torch.tensor(budgets, device=value.device)[:, None].expand_as(counts)
        if not torch.equal(counts, expected):
            raise RuntimeError(f"V5.2 {mode} failed exact budget validation")
        if not bool((value[2] & ~value[1]).sum() == 0) or not bool((value[1] & ~value[0]).sum() == 0):
            raise RuntimeError(f"V5.2 {mode} failed nested subset validation")

    retention_rows = []
    block_to_group = {
        block: group for group, (start, end) in enumerate(block_groups) for block in range(start, end + 1)
    }
    for mode, mode_masks in masks.items():
        for branch_index, branch in enumerate(branches):
            for block_index, block in enumerate(blocks):
                group = block_to_group[block]
                for frame in range(frames):
                    values = raw[branch_index, block_index, frame]
                    mass = float(values.sum())
                    retained = float(values[mode_masks[group, frame]].sum())
                    retention_rows.append(
                        {
                            "mode": mode,
                            "branch": branch,
                            "block": block,
                            "group": group,
                            "frame": frame + 1,
                            "raw_future_frame_mass": mass,
                            "retained_mass": retained,
                            "attention_mass_retention": retained / max(mass, float(eps)),
                        }
                    )

    return {
        "raw_profiles": raw,
        "branches": branches,
        "blocks": blocks,
        "block_mass": block_mass,
        "block_entropy": block_entropy,
        "block_quality": block_quality,
        "core_blocks": [blocks[int(index)] for index in core_block_indices],
        "core_weights": core_weights,
        "core_scores": core_scores,
        "core_masks": core_masks,
        "group_raw_scores": group_raw_scores,
        "v51_normalized": v51_normalized,
        "v51_smoothed": v51_smoothed,
        "v51_depth_scores": v51_depth_scores,
        "stable_mean": mean,
        "stable_std": std,
        "stable_cv": cv,
        "stable_scores": stable_scores,
        "stable_masks": stable_masks,
        "adaptive_scores": adaptive_scores,
        "execution_masks": masks,
        "category_labels": labels,
        "replacement_rows": replacement_rows,
        "retention_rows": retention_rows,
    }


class V52MotionCoreStableAdaptiveController(GroupedTemporalClosedROISparseController):
    """Opt-in A-D V5.2 controller; sparse execution itself is inherited."""

    def __init__(
        self,
        *,
        ablation_mode: str,
        stable_budgets: Sequence[int] = DEFAULT_STABLE_BUDGETS,
        core_block_range: tuple[int, int] = DEFAULT_CORE_BLOCK_RANGE,
        core_block_count: int = DEFAULT_CORE_BLOCK_COUNT,
        core_token_budget: int = DEFAULT_CORE_TOKEN_BUDGET,
        stable_cv_penalty: float = DEFAULT_STABLE_CV_PENALTY,
        replacement_relative_threshold: float = DEFAULT_REPLACEMENT_RELATIVE_THRESHOLD,
        max_replacements: int = DEFAULT_MAX_REPLACEMENTS,
        **kwargs: Any,
    ) -> None:
        mode = str(ablation_mode).lower()
        if mode not in ABLATION_MODES:
            raise ValueError(f"Unknown V5.2 ablation mode {ablation_mode!r}")
        super().__init__(**kwargs)
        self.ablation_mode = mode
        self.stable_budgets = tuple(int(value) for value in stable_budgets)
        self.core_block_range = tuple(int(value) for value in core_block_range)
        self.core_block_count = int(core_block_count)
        self.core_token_budget = int(core_token_budget)
        self.stable_cv_penalty = float(stable_cv_penalty)
        self.replacement_relative_threshold = float(replacement_relative_threshold)
        self.max_replacements = int(max_replacements)
        self.v52_plan: dict[str, Any] | None = None

    def _capture_step0_profile(self, **kwargs: Any) -> None:
        layer_index = int(kwargs["layer_index"])
        if self._current is None or self._layout is None or self._profile_callback_block is None:
            raise RuntimeError("V5.2 profile callback has no live context")
        if layer_index != self._profile_callback_block:
            raise RuntimeError(f"V5.2 profile expected B{self._profile_callback_block}, got B{layer_index}")
        raw = (
            action_aligned_future_raw_profiles(
                torch=self.torch,
                q_gen=kwargs["q_gen"],
                k_ar=kwargs["k_ar"],
                k_gen=kwargs["k_gen"],
                scaling=float(kwargs["scaling"]),
                token_layout=self._layout,
            )
            .detach()
            .cpu()
        )
        self._profile_tensors.append(raw)
        self.profile_records.append({"branch": str(self._current["branch"]), "block": layer_index, "profiles": raw})
        for latent in range(1, 9):
            values = raw[latent - 1]
            normalized = values / values.sum().clamp_min(self.torch.finfo(values.dtype).eps)
            entropy = float(-(normalized * normalized.clamp_min(1e-12).log()).sum() / math.log(values.numel()))
            self._profile_rows.append(
                {
                    **self._current,
                    "block": layer_index,
                    "latent": latent,
                    "raw_mass": float(values.sum()),
                    "raw_max": float(values.max()),
                    "normalized_spatial_entropy": entropy,
                }
            )

    def _finalize_roi_if_ready(self) -> None:
        if self.execution_masks is not None or self._current is None:
            return
        last_branch = "conditional" if self.guidance == 1.0 else "unconditional"
        if int(self._current["step"]) != 0 or str(self._current["branch"]) != last_branch:
            return
        expected = (1 if self.guidance == 1.0 else 2) * sum(end - start + 1 for start, end in self.block_groups)
        if len(self.profile_records) != expected:
            raise RuntimeError(f"Expected {expected} V5.2 profiles, got {len(self.profile_records)}")
        self.v52_plan = build_v52_ablation_plan(
            torch=self.torch,
            profile_records=self.profile_records,
            block_groups=self.block_groups,
            token_budgets=self.token_budgets,
            stable_budgets=self.stable_budgets,
            core_block_range=self.core_block_range,
            core_block_count=self.core_block_count,
            core_token_budget=self.core_token_budget,
            stable_cv_penalty=self.stable_cv_penalty,
            replacement_relative_threshold=self.replacement_relative_threshold,
            max_replacements=self.max_replacements,
            temporal_weights=self.temporal_weights,
            depth_lookahead_decay=self.depth_lookahead_decay,
        )
        self.normalized_scores = self.v52_plan["v51_normalized"]
        self.smoothed_scores = self.v52_plan["v51_smoothed"]
        self.depth_scores = self.v52_plan["v51_depth_scores"]
        self.execution_masks = self.v52_plan["execution_masks"][self.ablation_mode]

    def finish(self) -> dict[str, Any]:
        summary = super().finish()
        if self.v52_plan is None or self.execution_masks is None:
            raise RuntimeError("V5.2 plan is unavailable")
        violations_32 = int((self.execution_masks[2] & ~self.execution_masks[1]).sum())
        violations_21 = int((self.execution_masks[1] & ~self.execution_masks[0]).sum())
        labels = self.v52_plan["category_labels"][self.ablation_mode]
        category_names = {1: "core", 2: "stable", 3: "adaptive_positive", 4: "budget_fill"}
        category_counts = {
            name: [[int((labels[group, frame] == value).sum()) for frame in range(8)] for group in range(3)]
            for value, name in category_names.items()
        }
        retention_by_group = []
        for group in range(len(self.block_groups)):
            values = [
                float(row["attention_mass_retention"])
                for row in self.v52_plan["retention_rows"]
                if row["mode"] == self.ablation_mode and int(row["group"]) == group
            ]
            retention_by_group.append(
                {"group": group, "mean": sum(values) / len(values), "min": min(values), "count": len(values)}
            )
        temporal_jaccard_by_group = []
        for group in range(len(self.block_groups)):
            values = []
            for frame in range(7):
                left = self.execution_masks[group, frame]
                right = self.execution_masks[group, frame + 1]
                union = int((left | right).sum())
                values.append(float((left & right).sum()) / union if union else 1.0)
            temporal_jaccard_by_group.append({"group": group, "mean": sum(values) / len(values), "values": values})
        replacement_totals = {
            key: sum(int(row[key]) for row in self.v52_plan["replacement_rows"])
            for key in (
                "forced_replacements",
                "threshold_proposed",
                "threshold_accepted",
                "threshold_rejected",
            )
        }
        summary.update(
            {
                "schema_version": 2,
                "strategy_version": VELOCITY_CACHE_STRATEGY_VERSION,
                "ablation_mode": self.ablation_mode,
                "experiment": "motion_core_stable_adaptive_nested_k80_velocity_cache",
                "token_budgets": list(self.token_budgets),
                "stable_budgets": list(self.stable_budgets),
                "core_blocks": self.v52_plan["core_blocks"],
                "core_token_budget": self.core_token_budget,
                "stable_cv_penalty": self.stable_cv_penalty,
                "replacement_relative_threshold": self.replacement_relative_threshold,
                "max_replacements": self.max_replacements,
                "subset_violation_g3_not_g2": violations_32,
                "subset_violation_g2_not_g1": violations_21,
                "category_counts": category_counts,
                "attention_mass_retention_by_group": retention_by_group,
                "temporal_jaccard_by_group": temporal_jaccard_by_group,
                "replacement_totals": replacement_totals,
                "raw_attention_used_for_core": True,
                "spatial_normalization_used_only_for_entropy": True,
                "all_finite": True,
            }
        )
        if self.output_dir is not None:
            artifact = {
                key: value for key, value in self.v52_plan.items() if key not in {"retention_rows", "replacement_rows"}
            }
            artifact["selected_ablation_mode"] = self.ablation_mode
            artifact["selected_execution_masks"] = self.execution_masks
            self.torch.save(artifact, self.output_dir / "step0_v5_2_motion_core_stable_adaptive.pt")
            with (self.output_dir / "attention_mass_retention.csv").open("w", newline="", encoding="utf-8") as handle:
                rows = self.v52_plan["retention_rows"]
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with (self.output_dir / "replacement_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                rows = self.v52_plan["replacement_rows"]
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (self.output_dir / "token_savings_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        return summary


V52VelocityCacheSampler = GroupedTemporalClosedVelocityCacheSampler
