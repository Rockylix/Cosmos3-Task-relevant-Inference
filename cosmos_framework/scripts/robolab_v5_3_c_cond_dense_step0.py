# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""C/K80 ablation with only the step-0 conditional CFG branch dense."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq
from cosmos_framework.scripts.robolab_grouped_temporal_closed_roi_velocity_cache import (
    expand_grouped_token_roi_to_velocity_grid,
)
from cosmos_framework.scripts.robolab_v5_2_motion_core_stable_adaptive_velocity_cache import (
    V52MotionCoreStableAdaptiveController,
)
from cosmos_framework.scripts.robolab_v5_3_acd_packed_kernel_velocity_cache import (
    V53OptimizedACDController,
)

STRATEGY_VERSION = "v5.3-c-cond-dense-step0"


def step0_unconditional_group(block: int, groups: Sequence[tuple[int, int]]) -> int:
    """Use G1 before its normal B4 start, then preserve the configured groups."""

    if block < int(groups[0][0]):
        return 0
    for index, (start, end) in enumerate(groups):
        if int(start) <= block <= int(end):
            return index
    raise ValueError(f"Block B{block} is outside the step-0 unconditional schedule")


def zero_cfg_delta_outside_future_roi(
    *, torch: Any, conditional: Any, unconditional: Any, future_roi: Any
) -> tuple[Any, int]:
    """Set unconditional future prediction to conditional outside ``future_roi``."""

    if conditional.shape != unconditional.shape or conditional.ndim not in (4, 5):
        raise ValueError("Conditional/unconditional vision tensors must have matching 4D/5D shapes")
    temporal_dim = conditional.ndim - 3
    if int(conditional.shape[temporal_dim]) != 9:
        raise ValueError("Expected L0..L8 on the temporal dimension")
    height, width = int(conditional.shape[-2]), int(conditional.shape[-1])
    if tuple(future_roi.shape) != (8, height, width):
        raise ValueError(f"Expected future ROI [8,{height},{width}], got {tuple(future_roi.shape)}")
    conditional_future = conditional.narrow(temporal_dim, 1, 8)
    unconditional_future = unconditional.narrow(temporal_dim, 1, 8)
    view_shape = [1] * (conditional_future.ndim - 3) + [8, height, width]
    background = (~future_roi.to(device=unconditional.device)).reshape(view_shape).expand_as(
        unconditional_future
    )
    replaced = unconditional.clone()
    replaced.narrow(temporal_dim, 1, 8).copy_(
        torch.where(background, conditional_future.to(unconditional), unconditional_future)
    )
    return replaced, int(background.sum())


class V53CConditionalDenseStep0Controller(V53OptimizedACDController):
    """Profile conditional step 0 densely and sparsify its unconditional peer."""

    def __init__(self, **kwargs: Any) -> None:
        if str(kwargs.get("ablation_mode", "c")).lower() != "c":
            raise ValueError("Conditional-dense step-0 experiment supports only C")
        super().__init__(**kwargs)
        self._step0_conditional_vision: list[Any] = []
        self._future_velocity_roi: Any | None = None
        self._cfg_zero_fill_elements = 0
        self._step0_unconditional_sparse_blocks = 0

    def _finalize_roi_if_ready(self) -> None:
        if self.execution_masks is not None or self._current is None:
            return
        if int(self._current["step"]) != 0 or str(self._current["branch"]) != "conditional":
            return
        self._materialize_profiles_once()
        guidance = self.guidance
        self.guidance = 1.0
        try:
            V52MotionCoreStableAdaptiveController._finalize_roi_if_ready(self)
        finally:
            self.guidance = guidance
        if self.execution_masks is None:
            raise RuntimeError("Conditional step-0 profiles did not produce C masks")

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
            raise RuntimeError("Conditional-dense step-0 controller has no live forward")
        step = int(self._current["step"])
        branch = str(self._current["branch"])
        if step != 0 or branch == "conditional":
            return super().run_layer(
                block=block,
                decoder_layer=decoder_layer,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                memory_value=memory_value,
                gen_only=gen_only,
            )
        if self.execution_masks is None or self._layout is None:
            raise RuntimeError("Step-0 unconditional branch started before conditional C mask finalization")
        if self._position_embeddings is None or self._active_original_positions is None or self._side_buffer is None:
            raise RuntimeError("Step-0 unconditional sparse stack is incomplete")

        group = step0_unconditional_group(block, self.block_groups)
        group_start = 0 if group == 0 else int(self.block_groups[group][0])
        if block == group_start:
            self._prepare_group_indexes(get_gen_seq(hidden_states).device)
            if group > 0:
                with self._nvtx(f"v53c/step0/unconditional/G{group}/commit"):
                    self._side_buffer.index_copy_(
                        0, self._active_original_positions, get_gen_seq(hidden_states)
                    )
                self._side_buffer_updates += 1
            with self._nvtx(f"v53c/step0/unconditional/G{group}/pack"):
                hidden_states, self._position_embeddings = self._slice_pack_and_rope(
                    hidden_states, self._group_local_positions[group]
                )
            self._active_original_positions = self._group_original_positions[group]

        with self._nvtx(f"v53c/step0/unconditional/B{block}/sparse"):
            output, lbl_metadata, kv_to_store = decoder_layer(
                hidden_states,
                attention_mask,
                self._position_embeddings,
                natten_metadata=None,
                memory_value=memory_value,
                gen_only=gen_only,
            )
        sparse_gen = get_gen_seq(output)
        retained = int(sparse_gen.shape[0])
        if retained != int(self._active_original_positions.numel()):
            raise RuntimeError("Step-0 unconditional sparse block output length changed")
        original_gen = int(self._layout["num_gen_tokens"])
        self._block_rows.append(
            {
                **self._current,
                "block": block,
                "mode": "step0_unconditional_conditional_mask_sparse",
                "execution_runtime": "v53_group_packed_sync_light",
                "block_group": group,
                "future_tokens_by_frame": [
                    int(mask.sum()) for mask in self.execution_masks[group]
                ],
                "gen_tokens_before": retained,
                "gen_tokens_after": retained,
                "saved_gen_vs_original": original_gen - retained,
                "gen_retained_ratio": retained / original_gen,
                "finite": self._finite(sparse_gen),
            }
        )
        self._step0_unconditional_sparse_blocks += 1
        return output, lbl_metadata, kv_to_store

    def _network_post_hook(
        self, module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any], output: Any
    ) -> Any:
        current = None if self._current is None else dict(self._current)
        modified = output
        if current is not None and int(current["step"]) == 0 and isinstance(output, dict):
            predictions = output.get("preds_vision")
            if predictions:
                if str(current["branch"]) == "conditional":
                    self._step0_conditional_vision = [value.detach().clone() for value in predictions]
                elif str(current["branch"]) == "unconditional":
                    if len(predictions) != len(self._step0_conditional_vision):
                        raise RuntimeError("Step-0 conditional/unconditional sample count differs")
                    if self._layout is None or self.execution_masks is None:
                        raise RuntimeError("Cannot zero-fill CFG delta without token layout and masks")
                    if self._future_velocity_roi is None:
                        _, token_h, token_w = self._layout["latent_shape_thw"]
                        sample = predictions[0]
                        self._future_velocity_roi = expand_grouped_token_roi_to_velocity_grid(
                            torch=self.torch,
                            frame_masks=self.execution_masks[-1],
                            token_grid=(int(token_h), int(token_w)),
                            velocity_grid=(int(sample.shape[-2]), int(sample.shape[-1])),
                        ).to(device=sample.device)
                    replaced = []
                    for conditional, unconditional in zip(
                        self._step0_conditional_vision, predictions, strict=True
                    ):
                        value, count = zero_cfg_delta_outside_future_roi(
                            torch=self.torch,
                            conditional=conditional,
                            unconditional=unconditional,
                            future_roi=self._future_velocity_roi,
                        )
                        replaced.append(value)
                        self._cfg_zero_fill_elements += count
                    modified = dict(output)
                    modified["preds_vision"] = replaced
        super()._network_post_hook(module, args, kwargs, modified)
        return modified if modified is not output else None

    def finish(self) -> dict[str, Any]:
        summary = super().finish()
        if self._step0_unconditional_sparse_blocks != len(self.layers):
            raise RuntimeError(
                "Expected every step-0 unconditional block to be sparse, got "
                f"{self._step0_unconditional_sparse_blocks}/{len(self.layers)}"
            )
        if not self._step0_conditional_vision or self._cfg_zero_fill_elements <= 0:
            raise RuntimeError("Step-0 conditional prediction/CFG zero-fill was not observed")
        summary.update(
            {
                "strategy_version": STRATEGY_VERSION,
                "experiment": "c_k80_conditional_only_dense_step0",
                "aggregation": "conditional step0 profiles only; future frames separate",
                "step0_all_blocks_dense": False,
                "step0_conditional_all_blocks_dense": True,
                "step0_dense_branches": ["conditional"],
                "step0_unconditional_sparse_blocks": self._step0_unconditional_sparse_blocks,
                "step0_unconditional_groups": [
                    {"group": 0, "start_block": 0, "end_block": 11},
                    {"group": 1, "start_block": 12, "end_block": 19},
                    {"group": 2, "start_block": 20, "end_block": 27},
                ],
                "conditional_profiles_only": True,
                "step0_background_cfg_delta": "zero",
                "cfg_zero_fill_elements": self._cfg_zero_fill_elements,
            }
        )
        return summary
