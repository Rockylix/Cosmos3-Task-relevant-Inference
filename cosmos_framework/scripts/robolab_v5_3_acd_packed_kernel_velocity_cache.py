# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Optimized A/C/D grouped-token execution for the V5.3 experiment.

The selected tokens and model arithmetic are identical to V5.2.  This module
only removes experiment overhead from the timed path:

* Step-0 profiles are transferred to CPU in one batch instead of once per
  block.
* Group-local indexes are built once per request and reused by every CFG/step
  stack.
* The full-layout side buffer is updated only at G1->G2 and G2->G3 boundaries,
  rather than after every sparse block.
* Per-block finite checks are optional; the benchmark validates terminal
  outputs after the synchronized generation interval.

The existing FlashAttention and decoder-layer implementations remain intact.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from cosmos_framework.data.generator.sequence_packing.runtime import (
    from_und_gen_splits,
    get_gen_seq,
    get_und_seq,
)
from cosmos_framework.scripts.robolab_grouped_temporal_closed_roi_velocity_cache import (
    expand_grouped_token_roi_to_velocity_grid,
    grouped_roi_original_positions,
)
from cosmos_framework.scripts.robolab_v5_2_motion_core_stable_adaptive_velocity_cache import (
    V52MotionCoreStableAdaptiveController,
    action_aligned_future_raw_profiles,
)

VELOCITY_CACHE_STRATEGY_VERSION = "v5.3"


def subset_local_positions_searchsorted(torch: Any, active_original: Any, target_original: Any) -> Any:
    """Map a sorted nested target into a sorted active sequence without ``isin``."""

    if active_original.ndim != 1 or target_original.ndim != 1:
        raise ValueError("active_original and target_original must be one-dimensional")
    local = torch.searchsorted(active_original, target_original)
    if int(target_original.numel()):
        if int(local.max()) >= int(active_original.numel()) or not torch.equal(
            active_original.index_select(0, local), target_original
        ):
            raise RuntimeError("Target token set is not a sorted subset of the active token set")
    return local


class V53OptimizedACDController(V52MotionCoreStableAdaptiveController):
    """V5.2 A/C/D masks with a synchronization-light grouped execution path."""

    def __init__(
        self,
        *,
        validate_intermediates: bool = False,
        enable_nvtx: bool = False,
        **kwargs: Any,
    ) -> None:
        mode = str(kwargs.get("ablation_mode", "")).lower()
        if mode not in {"a", "c", "d"}:
            raise ValueError("V5.3 optimized controller supports only A, C, and D")
        super().__init__(**kwargs)
        self.validate_intermediates = bool(validate_intermediates)
        self.enable_nvtx = bool(enable_nvtx)
        self._profiles_materialized = False
        self._group_original_positions: list[Any] = []
        self._group_local_positions: list[Any] = []
        self._index_device: Any | None = None
        self._deferred_profile_contexts: list[dict[str, Any]] = []
        self._side_buffer_updates = 0

    @contextmanager
    def _nvtx(self, label: str) -> Iterator[None]:
        enabled = self.enable_nvtx and self.torch.cuda.is_available()
        if enabled:
            self.torch.cuda.nvtx.range_push(label)
        try:
            yield
        finally:
            if enabled:
                self.torch.cuda.nvtx.range_pop()

    def _capture_step0_profile(self, **kwargs: Any) -> None:
        layer_index = int(kwargs["layer_index"])
        if self._current is None or self._layout is None or self._profile_callback_block is None:
            raise RuntimeError("V5.3 profile callback has no live context")
        if layer_index != self._profile_callback_block:
            raise RuntimeError(f"V5.3 profile expected B{self._profile_callback_block}, got B{layer_index}")
        raw = action_aligned_future_raw_profiles(
            torch=self.torch,
            q_gen=kwargs["q_gen"],
            k_ar=kwargs["k_ar"],
            k_gen=kwargs["k_gen"],
            scaling=float(kwargs["scaling"]),
            token_layout=self._layout,
            validate=self.validate_intermediates,
        ).detach()
        self._profile_tensors.append(raw)
        self.profile_records.append({"branch": str(self._current["branch"]), "block": layer_index, "profiles": raw})
        self._deferred_profile_contexts.append(dict(self._current))

    def _materialize_profiles_once(self) -> None:
        if self._profiles_materialized:
            return
        if not self.profile_records:
            raise RuntimeError("No V5.3 profiles were captured")
        # One synchronized D2H transfer replaces one transfer per profiled block.
        stacked = self.torch.stack([record["profiles"] for record in self.profile_records]).cpu()
        materialized_records = []
        self._profile_rows.clear()
        for index, (record, context) in enumerate(
            zip(self.profile_records, self._deferred_profile_contexts, strict=True)
        ):
            raw = stacked[index]
            materialized_records.append(
                {"branch": str(record["branch"]), "block": int(record["block"]), "profiles": raw}
            )
            for latent in range(1, 9):
                values = raw[latent - 1]
                normalized = values / values.sum().clamp_min(self.torch.finfo(values.dtype).eps)
                entropy = float(-(normalized * normalized.clamp_min(1e-12).log()).sum() / math.log(values.numel()))
                self._profile_rows.append(
                    {
                        **context,
                        "block": int(record["block"]),
                        "latent": latent,
                        "raw_mass": float(values.sum()),
                        "raw_max": float(values.max()),
                        "normalized_spatial_entropy": entropy,
                    }
                )
        self._profile_tensors = [stacked[index] for index in range(int(stacked.shape[0]))]
        self.profile_records = materialized_records
        self._profiles_materialized = True

    def _finalize_roi_if_ready(self) -> None:
        if self.execution_masks is not None or self._current is None:
            return
        last_branch = "conditional" if self.guidance == 1.0 else "unconditional"
        if int(self._current["step"]) != 0 or str(self._current["branch"]) != last_branch:
            return
        self._materialize_profiles_once()
        super()._finalize_roi_if_ready()

    def _prepare_group_indexes(self, device: Any) -> None:
        if self._group_original_positions:
            if device != self._index_device:
                raise RuntimeError(f"GEN token device changed from {self._index_device} to {device}")
            return
        if self.execution_masks is None or self._layout is None:
            raise RuntimeError("Cannot prepare V5.3 indexes before mask finalization")
        cpu = self.torch.device("cpu")
        active = self.torch.arange(int(self._layout["num_gen_tokens"]), dtype=self.torch.long, device=cpu)
        originals_cpu = []
        locals_cpu = []
        for group in range(len(self.block_groups)):
            target = grouped_roi_original_positions(
                self.torch,
                self._layout,
                self.execution_masks[group].cpu(),
                cpu,
            )
            local = subset_local_positions_searchsorted(self.torch, active, target)
            originals_cpu.append(target)
            locals_cpu.append(local)
            active = target
        self._group_original_positions = [value.to(device=device) for value in originals_cpu]
        self._group_local_positions = [value.to(device=device) for value in locals_cpu]
        self._index_device = device

    def _finite(self, value: Any) -> bool:
        return bool(self.torch.isfinite(value).all()) if self.validate_intermediates else True

    def _run_dense_layer(
        self,
        *,
        block: int,
        decoder_layer: Any,
        hidden_states: Any,
        attention_mask: Any,
        memory_value: Any,
        gen_only: bool,
    ) -> tuple[Any, dict[str, Any], Any]:
        assert self._current is not None and self._position_embeddings is not None
        step = int(self._current["step"])
        attention = decoder_layer.self_attn
        capture = step == 0 and block >= self.first_sparse_block
        if capture:
            if attention._attention_stats_capture_callback is not None:
                raise RuntimeError(f"B{block} attention callback is occupied")
            self._profile_callback_block = block
            attention._attention_stats_capture_callback = self._capture_step0_profile
        label = f"v53/step{step}/{self._current['branch']}/B{block}/dense"
        try:
            with self._nvtx(label):
                output, lbl_metadata, kv_to_store = decoder_layer(
                    hidden_states,
                    attention_mask,
                    self._position_embeddings,
                    natten_metadata=None,
                    memory_value=memory_value,
                    gen_only=gen_only,
                )
        finally:
            if capture:
                attention._attention_stats_capture_callback = None
                self._profile_callback_block = None
        full_gen = get_gen_seq(output)
        # Later sparse stacks need the complete B3 state as their restoration base.
        if step > 0 and block == self.first_sparse_block - 1:
            self._side_buffer = full_gen.detach().clone()
            self._side_buffer_updates += 1
        self._block_rows.append(
            {
                **self._current,
                "block": block,
                "mode": "step0_profile_dense" if step == 0 else "dense_head",
                "gen_tokens_before": int(full_gen.shape[0]),
                "gen_tokens_after": int(full_gen.shape[0]),
                "saved_gen_vs_original": 0,
                "gen_retained_ratio": 1.0,
                "finite": self._finite(full_gen),
            }
        )
        return output, lbl_metadata, kv_to_store

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
        if not self.stack_active or self._current is None or self._layout is None:
            raise RuntimeError("V5.3 run_layer called without an active stack")
        if self._position_embeddings is None or self._active_original_positions is None or self._side_buffer is None:
            raise RuntimeError("V5.3 sparse stack state is incomplete")
        step = int(self._current["step"])
        if step == 0 or block < self.first_sparse_block:
            return self._run_dense_layer(
                block=block,
                decoder_layer=decoder_layer,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                memory_value=memory_value,
                gen_only=gen_only,
            )
        if self.execution_masks is None:
            raise RuntimeError("Later denoise step started before V5.3 masks were finalized")

        group = self._group_index(block)
        if block == self.block_groups[group][0]:
            self._prepare_group_indexes(get_gen_seq(hidden_states).device)
            if group > 0:
                # Commit the previous group's terminal state once before tokens leave.
                with self._nvtx(f"v53/step{step}/{self._current['branch']}/G{group}/commit"):
                    self._side_buffer.index_copy_(
                        0,
                        self._active_original_positions,
                        get_gen_seq(hidden_states),
                    )
                self._side_buffer_updates += 1
            with self._nvtx(f"v53/step{step}/{self._current['branch']}/G{group}/pack"):
                hidden_states, self._position_embeddings = self._slice_pack_and_rope(
                    hidden_states,
                    self._group_local_positions[group],
                )
            self._active_original_positions = self._group_original_positions[group]

        label = f"v53/step{step}/{self._current['branch']}/B{block}/sparse"
        with self._nvtx(label):
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
            raise RuntimeError("V5.3 sparse block output length changed")
        original_gen = int(self._layout["num_gen_tokens"])
        frame_counts = [int(mask.sum()) for mask in self.execution_masks[group]]
        retained = int(sparse_gen.shape[0])
        self._block_rows.append(
            {
                **self._current,
                "block": block,
                "mode": "grouped_score_smoothed_budgeted_sparse",
                "execution_runtime": "v53_group_packed_sync_light",
                "block_group": group,
                "future_tokens_by_frame": frame_counts,
                "gen_tokens_before": retained,
                "gen_tokens_after": retained,
                "saved_gen_vs_original": original_gen - retained,
                "gen_retained_ratio": retained / original_gen,
                "finite": self._finite(sparse_gen),
            }
        )
        return output, lbl_metadata, kv_to_store

    def end_stack(self, hidden_states: Any) -> Any:
        if not self.stack_active or self._original_pack is None or self._current is None:
            raise RuntimeError("end_stack called without an active V5.3 stack")
        assert self._active_original_positions is not None and self._side_buffer is not None
        active_hidden = get_gen_seq(hidden_states)
        full_gen = int(self._layout["num_gen_tokens"]) if self._layout is not None else -1
        if int(active_hidden.shape[0]) == full_gen:
            restored = hidden_states
        else:
            with self._nvtx(f"v53/step{self._current['step']}/{self._current['branch']}/restore"):
                restored_gen = self._side_buffer.clone()
                restored_gen.index_copy_(0, self._active_original_positions, active_hidden)
                restored = from_und_gen_splits(get_und_seq(hidden_states), restored_gen, self._original_pack)
            self._side_buffer_updates += 1
        if self.validate_intermediates and not bool(self.torch.isfinite(get_gen_seq(restored)).all()):
            raise RuntimeError("V5.3 restored terminal hidden contains NaN/Inf")
        self._completed_stacks += 1
        self._finalize_roi_if_ready()
        self.abort_stack()
        return restored

    def finish(self) -> dict[str, Any]:
        summary = super().finish()
        summary.update(
            {
                "strategy_version": VELOCITY_CACHE_STRATEGY_VERSION,
                "experiment": "acd_budget_group_packed_kernel_velocity_cache",
                "execution_runtime": "group_packed_sync_light",
                "intermediate_validation": self.validate_intermediates,
                "side_buffer_updates": self._side_buffer_updates,
                "group_indexes_cached": len(self._group_local_positions),
            }
        )
        if self.output_dir is not None:
            (Path(self.output_dir) / "token_savings_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return summary


class V53VelocityCacheSampler:
    """Production-like V5.2 velocity merge without analysis-time reductions."""

    def __init__(self, inner: Any, controller: V53OptimizedACDController) -> None:
        self.inner = inner
        self.controller = controller
        self.cached_step0_vision: list[Any] = []
        self._velocity_roi: Any | None = None
        self.completed_steps = 0

    def __call__(self, velocity_fn: Any, initial_noise: Any, **kwargs: Any) -> Any:
        self.cached_step0_vision.clear()
        self._velocity_roi = None
        self.completed_steps = 0

        def cached(noise_x: Any, timestep: Any) -> Any:
            current = velocity_fn(noise_x, timestep)
            if self.controller.vision_shape is None or self.controller.execution_masks is None:
                raise RuntimeError("V5.3 controller did not expose vision shape/masks")
            vision_numel = math.prod(self.controller.vision_shape)
            output = []
            for sample, flat_velocity in enumerate(current):
                vision = flat_velocity[:vision_numel].reshape(self.controller.vision_shape)
                if self.completed_steps == 0:
                    self.cached_step0_vision.append(vision.detach().clone())
                    merged = vision
                else:
                    if self.controller._layout is None:
                        raise RuntimeError("V5.3 controller token layout is unavailable")
                    if self._velocity_roi is None:
                        _, token_h, token_w = self.controller._layout["latent_shape_thw"]
                        self._velocity_roi = expand_grouped_token_roi_to_velocity_grid(
                            torch=self.controller.torch,
                            frame_masks=self.controller.execution_masks[-1],
                            token_grid=(int(token_h), int(token_w)),
                            velocity_grid=(int(vision.shape[-2]), int(vision.shape[-1])),
                        ).to(device=vision.device)
                    cached_vision = self.cached_step0_vision[sample].to(
                        device=vision.device,
                        dtype=vision.dtype,
                    )
                    temporal_dim = vision.ndim - 3
                    current_future = vision.narrow(temporal_dim, 1, 8)
                    cached_future = cached_vision.narrow(temporal_dim, 1, 8)
                    view_shape = [1] * (current_future.ndim - 3) + [
                        8,
                        int(vision.shape[-2]),
                        int(vision.shape[-1]),
                    ]
                    background = (~self._velocity_roi).reshape(view_shape).expand_as(current_future)
                    merged = vision.clone()
                    merged.narrow(temporal_dim, 1, 8).copy_(
                        self.controller.torch.where(background, cached_future, current_future)
                    )
                output.append(self.controller.torch.cat((merged.reshape(-1), flat_velocity[vision_numel:]), dim=0))
            self.completed_steps += 1
            return output

        result = self.inner(cached, initial_noise, **kwargs)
        expected = int(kwargs.get("num_steps", self.controller.num_steps))
        if self.completed_steps != expected:
            raise RuntimeError(f"Expected {expected} V5.3 guided velocity calls, got {self.completed_steps}")
        return result
