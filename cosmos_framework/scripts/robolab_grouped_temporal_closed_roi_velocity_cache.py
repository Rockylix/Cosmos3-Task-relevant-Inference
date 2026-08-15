# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Grouped, temporally closed ROI sparsity with a step-0 velocity cache.

Step 0 is dense.  Its post-RoPE Action-Q/Future-K profiles produce one mask
per block group and future latent.  A radius-one temporal closure discourages
adjacent latent frames from switching abruptly between current and cached
denoising.  Masks are nested over depth, so a token may leave the active
sequence at B12 or B20 but can never re-enter without the missing block state.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq
from cosmos_framework.scripts.robolab_action_attention_mass90_intervention import minimum_mass_mask
from cosmos_framework.scripts.robolab_step0_fixed_roi_velocity_cache import (
    GuidedBackgroundVelocityCacheSampler,
    Step0FixedROISparseController,
    _flat_metrics,
    action_aligned_future_spatial_profiles,
    expand_token_roi_to_velocity_grid,
)

VELOCITY_CACHE_STRATEGY_VERSION = "v5"
DEFAULT_BLOCK_GROUPS = ((4, 11), (12, 19), (20, 27))


def build_grouped_temporal_closed_masks(
    *,
    torch: Any,
    profile_records: Sequence[Mapping[str, Any]],
    block_groups: Sequence[tuple[int, int]],
    threshold: float,
) -> tuple[Any, Any, Any, Any]:
    """Build raw, temporal-closed, and depth-nested masks.

    Each record contains a ``profiles`` tensor shaped ``[8, spatial]``.  Scores
    are max-aggregated only over CFG branches and blocks inside one block group;
    future frames remain separate.  Returned tensors are all
    ``[block_group, future_frame, spatial]``.
    """

    if not profile_records:
        raise ValueError("At least one step-0 profile record is required")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0,1]")
    first = profile_records[0]["profiles"]
    if first.ndim != 2 or int(first.shape[0]) != 8:
        raise ValueError(f"Expected profile shape [8,spatial], got {tuple(first.shape)}")
    spatial = int(first.shape[1])
    scores = torch.empty((len(block_groups), 8, spatial), dtype=first.dtype, device=first.device)
    for group_index, (start, end) in enumerate(block_groups):
        if start > end:
            raise ValueError(f"Invalid block group {(start, end)}")
        members = [record["profiles"] for record in profile_records if start <= int(record["block"]) <= end]
        if not members:
            raise RuntimeError(f"No step-0 profiles were captured for block group {(start, end)}")
        if any(tuple(item.shape) != (8, spatial) for item in members):
            raise RuntimeError("Step-0 profile geometry changed inside a block group")
        scores[group_index] = torch.stack(members).amax(dim=0)

    raw_masks = minimum_mass_mask(scores.reshape(-1, spatial), threshold).reshape_as(scores)

    closed_masks = raw_masks.clone()
    for frame in range(8):
        lo = max(0, frame - 1)
        hi = min(8, frame + 2)
        closed_masks[:, frame] = raw_masks[:, lo:hi].any(dim=1)

    execution_masks = closed_masks.clone()
    running = torch.zeros_like(closed_masks[0])
    for group_index in reversed(range(len(block_groups))):
        running = running | closed_masks[group_index]
        execution_masks[group_index] = running

    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError("Grouped ROI scores contain NaN/Inf")
    return scores, raw_masks, closed_masks, execution_masks


def grouped_roi_original_positions(
    torch: Any,
    token_layout: Mapping[str, Any],
    frame_masks: Any,
    device: Any,
) -> Any:
    """Keep L0/action and an independently selected set for each future frame."""

    spatial = len(token_layout["latent_positions"]["L1"])
    if tuple(frame_masks.shape) != (8, spatial):
        raise ValueError(f"Expected frame masks [8,{spatial}], got {tuple(frame_masks.shape)}")
    parts = [torch.tensor(token_layout["latent_positions"]["L0"], dtype=torch.long, device=device)]
    for latent in range(1, 9):
        frame = torch.tensor(token_layout["latent_positions"][f"L{latent}"], dtype=torch.long, device=device)
        selected_spatial = torch.nonzero(frame_masks[latent - 1].to(device=device), as_tuple=False).flatten()
        parts.append(frame.index_select(0, selected_spatial))
    parts.append(torch.tensor(token_layout["action_positions"], dtype=torch.long, device=device))
    return torch.cat(parts).sort().values


def nested_subset_local_positions(torch: Any, active_original: Any, target_original: Any) -> Any:
    """Return local indexes for a sorted target that must be a subset of active."""

    membership = torch.isin(active_original, target_original)
    selected_local = torch.nonzero(membership, as_tuple=False).flatten()
    selected_original = active_original.index_select(0, selected_local)
    if int(selected_original.numel()) != int(target_original.numel()) or not torch.equal(
        selected_original, target_original
    ):
        missing = target_original[~torch.isin(target_original, active_original)]
        raise RuntimeError(
            "Grouped ROI attempted to re-enter tokens without continuous hidden state: "
            f"missing={missing.detach().cpu().tolist()[:16]}"
        )
    return selected_local


def expand_grouped_token_roi_to_velocity_grid(
    *, torch: Any, frame_masks: Any, token_grid: tuple[int, int], velocity_grid: tuple[int, int]
) -> Any:
    """Expand eight frame-local token masks to ``[8, velocity_h, velocity_w]``."""

    token_h, token_w = map(int, token_grid)
    if tuple(frame_masks.shape) != (8, token_h * token_w):
        raise ValueError(f"Expected [8,{token_h * token_w}] masks, got {tuple(frame_masks.shape)}")
    return torch.stack(
        [
            expand_token_roi_to_velocity_grid(
                torch=torch,
                roi_mask=frame_masks[frame],
                token_grid=token_grid,
                velocity_grid=velocity_grid,
            )
            for frame in range(8)
        ]
    )


def merge_grouped_cached_background_velocity(
    *, torch: Any, current: Any, cached_step0: Any, frame_roi_mask: Any
) -> tuple[Any, dict[str, float]]:
    """Use current frame-local ROI and step-0 background vision velocity."""

    if current.shape != cached_step0.shape or current.ndim not in (4, 5):
        raise ValueError(
            "Current/cached vision velocities must be matching [C,T,H,W] or [B,C,T,H,W], "
            f"got {tuple(current.shape)} and {tuple(cached_step0.shape)}"
        )
    temporal_dim = current.ndim - 3
    height, width = int(current.shape[-2]), int(current.shape[-1])
    if int(current.shape[temporal_dim]) != 9 or tuple(frame_roi_mask.shape) != (8, height, width):
        raise ValueError("Expected L0+L1..L8 and a frame-local [8,H,W] ROI mask")
    current_future = current.narrow(temporal_dim, 1, 8)
    cached_future = cached_step0.narrow(temporal_dim, 1, 8).to(device=current.device, dtype=current.dtype)
    view_shape = [1] * (current_future.ndim - 3) + [8, height, width]
    background = (~frame_roi_mask.to(device=current.device)).reshape(view_shape).expand_as(current_future)
    merged = current.clone()
    merged.narrow(temporal_dim, 1, 8).copy_(torch.where(background, cached_future, current_future))
    metrics = _flat_metrics(torch, current_future[background], cached_future[background])
    metrics["background_elements"] = int(background.sum())
    if not bool(torch.isfinite(merged).all()):
        raise RuntimeError("Grouped cached velocity contains NaN/Inf")
    return merged, metrics


class GroupedTemporalClosedROISparseController(Step0FixedROISparseController):
    """Dense step-0 profile followed by nested grouped/frame-local sparsity."""

    def __init__(
        self,
        *,
        torch: Any,
        net: Any,
        guidance: float,
        num_steps: int,
        threshold: float = 0.9,
        block_groups: Sequence[tuple[int, int]] = DEFAULT_BLOCK_GROUPS,
        output_dir: Path | None = None,
    ) -> None:
        groups = tuple((int(start), int(end)) for start, end in block_groups)
        layers = list(net.language_model.model.layers)
        if not groups or groups[0][0] <= 0 or groups[-1][1] >= len(layers):
            raise ValueError("Block groups must leave a non-empty dense head and fit the decoder")
        if any(groups[index][1] + 1 != groups[index + 1][0] for index in range(len(groups) - 1)):
            raise ValueError("Block groups must be contiguous")
        super().__init__(
            torch=torch,
            net=net,
            guidance=guidance,
            num_steps=num_steps,
            threshold=threshold,
            first_sparse_block=groups[0][0],
            output_dir=output_dir,
        )
        self.block_groups = groups
        self.profile_records: list[dict[str, Any]] = []
        self.grouped_scores: Any | None = None
        self.raw_masks: Any | None = None
        self.closed_masks: Any | None = None
        self.execution_masks: Any | None = None

    def _capture_step0_profile(
        self,
        *,
        layer_index: int,
        q_gen: Any,
        k_ar: Any,
        k_gen: Any,
        v_ar: Any,
        v_gen: Any,
        attn_output_gen: Any,
        scaling: float,
    ) -> None:
        del v_ar, v_gen, attn_output_gen
        if self._current is None or self._layout is None or self._profile_callback_block is None:
            raise RuntimeError("V5 profile callback has no live context")
        if layer_index != self._profile_callback_block:
            raise RuntimeError(f"V5 profile callback expected B{self._profile_callback_block}, got B{layer_index}")
        profiles = (
            action_aligned_future_spatial_profiles(
                torch=self.torch,
                q_gen=q_gen,
                k_ar=k_ar,
                k_gen=k_gen,
                scaling=scaling,
                token_layout=self._layout,
            )
            .detach()
            .cpu()
        )
        self._profile_tensors.append(profiles)
        self.profile_records.append(
            {
                "branch": str(self._current["branch"]),
                "block": int(layer_index),
                "profiles": profiles,
            }
        )
        for latent in range(1, 9):
            self._profile_rows.append(
                {
                    **self._current,
                    "block": int(layer_index),
                    "latent": latent,
                    "profile_sum": float(profiles[latent - 1].sum()),
                    "profile_max": float(profiles[latent - 1].max()),
                }
            )

    def _finalize_roi_if_ready(self) -> None:
        if self.execution_masks is not None or self._current is None:
            return
        last_profile_branch = "conditional" if self.guidance == 1.0 else "unconditional"
        if int(self._current["step"]) != 0 or str(self._current["branch"]) != last_profile_branch:
            return
        expected = (1 if self.guidance == 1.0 else 2) * sum(end - start + 1 for start, end in self.block_groups)
        if len(self.profile_records) != expected:
            raise RuntimeError(f"Expected {expected} V5 step-0 profiles, got {len(self.profile_records)}")
        self.grouped_scores, self.raw_masks, self.closed_masks, self.execution_masks = (
            build_grouped_temporal_closed_masks(
                torch=self.torch,
                profile_records=self.profile_records,
                block_groups=self.block_groups,
                threshold=self.threshold,
            )
        )

    def _group_index(self, block: int) -> int:
        for index, (start, end) in enumerate(self.block_groups):
            if start <= block <= end:
                return index
        raise RuntimeError(f"Block B{block} is outside configured sparse groups")

    def run_layer(
        self,
        *,
        block: int,
        decoder_layer: Any,
        hidden_states: Any,
        attention_mask: Any,
        memory_value: Any,
        gen_only: bool,
    ) -> tuple[Any, dict[str, Any], Any]:
        if self._current is None:
            raise RuntimeError("V5 run_layer has no active forward")
        step = int(self._current["step"])
        if step == 0 or block < self.first_sparse_block:
            return super().run_layer(
                block=block,
                decoder_layer=decoder_layer,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                memory_value=memory_value,
                gen_only=gen_only,
            )
        if self.execution_masks is None or self._layout is None:
            raise RuntimeError("Later denoise step started before V5 masks were finalized")
        if self._position_embeddings is None or self._active_original_positions is None or self._side_buffer is None:
            raise RuntimeError("V5 sparse stack state is incomplete")

        group_index = self._group_index(block)
        group_start = self.block_groups[group_index][0]
        if block == group_start:
            target_original = grouped_roi_original_positions(
                self.torch,
                self._layout,
                self.execution_masks[group_index],
                get_gen_seq(hidden_states).device,
            )
            selected_local = nested_subset_local_positions(
                self.torch,
                self._active_original_positions,
                target_original,
            )
            hidden_states, self._position_embeddings = self._slice_pack_and_rope(hidden_states, selected_local)
            self._active_original_positions = target_original

        output, lbl_metadata, kv_to_store = decoder_layer(
            hidden_states,
            attention_mask,
            self._position_embeddings,
            natten_metadata=None,
            memory_value=memory_value,
            gen_only=gen_only,
        )
        sparse_gen = get_gen_seq(output)
        if int(sparse_gen.shape[0]) != int(self._active_original_positions.numel()):
            raise RuntimeError("V5 sparse block output length changed")
        self._side_buffer.index_copy_(0, self._active_original_positions, sparse_gen)
        original_gen = int(self._layout["num_gen_tokens"])
        frame_counts = [int(mask.sum()) for mask in self.execution_masks[group_index]]
        retained = int(sparse_gen.shape[0])
        self._block_rows.append(
            {
                **self._current,
                "block": block,
                "mode": "grouped_temporal_closed_sparse",
                "block_group": group_index,
                "future_tokens_by_frame": frame_counts,
                "gen_tokens_before": retained,
                "gen_tokens_after": retained,
                "saved_gen_vs_original": original_gen - retained,
                "gen_retained_ratio": retained / original_gen,
                "finite": bool(self.torch.isfinite(sparse_gen).all()),
            }
        )
        return output, lbl_metadata, kv_to_store

    def finish(self) -> dict[str, Any]:
        expected_stacks = self.num_steps if self.guidance == 1.0 else 2 * self.num_steps
        expected_blocks = expected_stacks * len(self.layers)
        if self.stack_active or any(
            item is None for item in (self.grouped_scores, self.raw_masks, self.closed_masks, self.execution_masks)
        ):
            raise RuntimeError("V5 experiment did not complete")
        if self._completed_stacks != expected_stacks or len(self._block_rows) != expected_blocks:
            raise RuntimeError(
                f"Incomplete V5 stacks={self._completed_stacks}/{expected_stacks}, "
                f"blocks={len(self._block_rows)}/{expected_blocks}"
            )
        if not all(bool(row["finite"]) for row in self._block_rows):
            raise RuntimeError("V5 sparse output contains NaN/Inf")
        assert self._layout is not None and self.execution_masks is not None
        full_gen = int(self._layout["num_gen_tokens"])
        group_rows = []
        for group_index, (start, end) in enumerate(self.block_groups):
            counts = [int(mask.sum()) for mask in self.execution_masks[group_index]]
            retained = len(self._layout["latent_positions"]["L0"]) + len(self._layout["action_positions"]) + sum(counts)
            group_rows.append(
                {
                    "group": group_index,
                    "start_block": start,
                    "end_block": end,
                    "future_tokens_by_frame": counts,
                    "future_tokens": sum(counts),
                    "gen_tokens": retained,
                    "saved_gen_tokens": full_gen - retained,
                    "gen_retained_ratio": retained / full_gen,
                }
            )
        sparse_rows = [row for row in self._block_rows if row["mode"] == "grouped_temporal_closed_sparse"]
        summary = {
            "schema_version": 1,
            "strategy_version": VELOCITY_CACHE_STRATEGY_VERSION,
            "experiment": "grouped_temporal_closed_roi_guided_background_velocity_cache",
            "threshold": self.threshold,
            "aggregation": "max over step0 CFG branch x blocks within group; future frames separate",
            "block_groups": [list(group) for group in self.block_groups],
            "temporal_closure_radius": 1,
            "depth_nested": True,
            "token_reentry_allowed": False,
            "step0_all_blocks_dense": True,
            "later_dense_blocks": list(range(self.first_sparse_block)),
            "cfg_branches_share_masks": True,
            "future_frames_have_separate_masks": True,
            "full_gen_tokens": full_gen,
            "block_group_token_savings": group_rows,
            "average_saved_gen_tokens_per_sparse_block": sum(row["saved_gen_vs_original"] for row in sparse_rows)
            / len(sparse_rows),
            "average_saved_gen_tokens_per_all_block_call": sum(row["saved_gen_vs_original"] for row in self._block_rows)
            / len(self._block_rows),
            "profile_count": len(self.profile_records),
            "vision_shape": list(self.vision_shape or ()),
            "all_finite": True,
        }
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.torch.save(
                {
                    "profiles": self.torch.stack(self._profile_tensors),
                    "grouped_scores": self.grouped_scores,
                    "raw_masks": self.raw_masks,
                    "closed_masks": self.closed_masks,
                    "execution_masks": self.execution_masks,
                },
                self.output_dir / "step0_grouped_temporal_closed_roi.pt",
            )
            with (self.output_dir / "step0_profile_rows.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self._profile_rows[0]))
                writer.writeheader()
                writer.writerows(self._profile_rows)
            with (self.output_dir / "block_token_savings.csv").open("w", newline="", encoding="utf-8") as handle:
                fieldnames = sorted({key for row in self._block_rows for key in row})
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._block_rows)
            (self.output_dir / "token_savings_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        return summary


class GroupedTemporalClosedVelocityCacheSampler(GuidedBackgroundVelocityCacheSampler):
    """Merge frame-local current ROI and cached step-0 background before UniPC."""

    controller: GroupedTemporalClosedROISparseController

    def __call__(self, velocity_fn: Any, initial_noise: Any, **kwargs: Any) -> Any:
        self.velocities.clear()
        self.timesteps.clear()
        self.cache_trace.clear()
        self.cached_step0_vision.clear()
        step = 0

        def cached(noise_x: Any, timestep: Any) -> Any:
            nonlocal step
            current = velocity_fn(noise_x, timestep)
            if self.controller.vision_shape is None or self.controller.execution_masks is None:
                raise RuntimeError("V5 controller did not expose vision shape/masks after guided velocity")
            vision_numel = 1
            for value in self.controller.vision_shape:
                vision_numel *= int(value)
            output = []
            step_record: dict[str, Any] = {
                "step": step,
                "timestep": float(timestep.reshape(-1)[0]),
                "background_source_step": 0,
            }
            for sample, flat_velocity in enumerate(current):
                vision = flat_velocity[:vision_numel].reshape(self.controller.vision_shape)
                if step == 0:
                    self.cached_step0_vision.append(vision.detach().clone())
                    merged = vision
                    cache_metrics = {
                        "cosine": 1.0,
                        "relative_l2": 0.0,
                        "mse": 0.0,
                        "max_absolute_error": 0.0,
                    }
                else:
                    if self.controller._layout is None:
                        raise RuntimeError("V5 controller token layout is unavailable")
                    _, token_h, token_w = self.controller._layout["latent_shape_thw"]
                    velocity_roi = expand_grouped_token_roi_to_velocity_grid(
                        torch=self.controller.torch,
                        frame_masks=self.controller.execution_masks[-1],
                        token_grid=(int(token_h), int(token_w)),
                        velocity_grid=(int(vision.shape[-2]), int(vision.shape[-1])),
                    )
                    merged, cache_metrics = merge_grouped_cached_background_velocity(
                        torch=self.controller.torch,
                        current=vision,
                        cached_step0=self.cached_step0_vision[sample],
                        frame_roi_mask=velocity_roi,
                    )
                restored = self.controller.torch.cat((merged.reshape(-1), flat_velocity[vision_numel:]), dim=0)
                output.append(restored)
                for key, value in cache_metrics.items():
                    step_record[f"sample_{sample}_{key}"] = value
                if self.reference_velocities is not None:
                    reference = self.reference_velocities[step][sample]
                    step_record[f"sample_{sample}_merged_vs_baseline"] = _flat_metrics(
                        self.controller.torch, reference, restored
                    )
                    step_record[f"sample_{sample}_premerge_vs_baseline"] = _flat_metrics(
                        self.controller.torch, reference, flat_velocity
                    )
            if not all(bool(self.controller.torch.isfinite(item).all()) for item in output):
                raise RuntimeError("V5 cached velocity output contains NaN/Inf")
            self.velocities.append([item.detach().clone() for item in output])
            self.timesteps.append(float(timestep.reshape(-1)[0]))
            self.cache_trace.append(step_record)
            step += 1
            return output

        result = self.inner(cached, initial_noise, **kwargs)
        expected = int(kwargs.get("num_steps", self.controller.num_steps))
        if step != expected:
            raise RuntimeError(f"Expected {expected} V5 guided velocity calls, got {step}")
        return result
