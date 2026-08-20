# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Step-0 profile, fixed future ROI, and guided background-velocity cache.

Step 0 is dense in every decoder block.  Its real post-RoPE Action-Q/Future-K
attention produces one request-local spatial ROI shared by L1..L8, both CFG
branches, all later denoise steps, and B4..B27.  Steps 1..N keep B0..B3 dense,
then make the whole remaining decoder stack consume a genuinely shorter GEN
SequencePack.  Before UniPC, later guided velocities use the current prediction
inside the ROI and step-0's guided velocity outside it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    from_und_gen_splits,
    get_gen_seq,
    get_und_seq,
)
from cosmos_framework.scripts.robolab_action_attention_mass90_intervention import (
    ActionAttentionMass90Controller,
    minimum_mass_mask,
    restore_terminal_hidden,
)


VELOCITY_CACHE_STRATEGY_VERSION = "v1"


def action_aligned_future_spatial_profiles(
    *,
    torch: Any,
    q_gen: Any,
    k_ar: Any,
    k_gen: Any,
    scaling: float,
    token_layout: Mapping[str, Any],
    eps: float = 1e-12,
) -> Any:
    """Return eight frame-local Action-aligned distributions, shape ``[8,S]``."""

    if q_gen.ndim != 3 or k_ar.ndim != 3 or k_gen.ndim != 3:
        raise ValueError("Expected Q/K [tokens,heads,head_dim]")
    if int(q_gen.shape[0]) != int(token_layout["num_gen_tokens"]):
        raise RuntimeError("Step-0 profile requires a complete GEN sequence")
    q_heads = int(q_gen.shape[1])
    kv_heads = int(k_gen.shape[1])
    if q_heads % kv_heads:
        raise RuntimeError(f"Invalid GQA geometry Hq={q_heads}, Hkv={kv_heads}")

    predicted = [query for query in token_layout["action_queries"] if query["query_role"] == "predicted"]
    horizon_to_position = {int(query["action_horizon"]): int(query["gen_position"]) for query in predicted}
    if sorted(horizon_to_position) != list(range(32)):
        raise RuntimeError("Expected predicted Action horizons 0..31")
    action_index = torch.tensor(
        [horizon_to_position[horizon] for horizon in range(32)],
        dtype=torch.long,
        device=q_gen.device,
    )
    q = q_gen.index_select(0, action_index).detach().float().permute(1, 0, 2).contiguous()
    k = torch.cat((k_ar, k_gen), dim=0).detach().float()
    k = k.repeat_interleave(q_heads // kv_heads, dim=1).permute(1, 0, 2).contiguous()
    probabilities = torch.softmax(torch.matmul(q, k.transpose(1, 2)) * float(scaling), dim=-1)
    head_mean = probabilities.mean(dim=0)
    if not bool(torch.isfinite(head_mean).all()):
        raise RuntimeError("Action attention profile contains NaN/Inf")

    num_ar = int(k_ar.shape[0])
    profiles = []
    for latent in range(1, 9):
        horizons = list(range(4 * (latent - 1), 4 * latent))
        positions = torch.tensor(
            token_layout["latent_positions"][f"L{latent}"],
            dtype=torch.long,
            device=q_gen.device,
        )
        weights = head_mean[horizons].index_select(-1, positions + num_ar).mean(dim=0)
        profiles.append(weights / weights.sum().clamp_min(eps))
    return torch.stack(profiles)


def aggregate_fixed_roi(torch: Any, profiles: Any, threshold: float) -> tuple[Any, Any]:
    """Max-pool profiles and return their minimum ``threshold``-mass mask."""

    if profiles.ndim != 3:
        raise ValueError(f"Expected [profiles,future,spatial], got {tuple(profiles.shape)}")
    if int(profiles.shape[1]) != 8:
        raise ValueError("Expected L1..L8 profiles")
    aggregate = profiles.amax(dim=(0, 1))
    mask = minimum_mass_mask(aggregate.unsqueeze(0), threshold).squeeze(0)
    if not int(mask.sum()) or int(mask.sum()) == int(mask.numel()):
        raise RuntimeError("Fixed ROI must leave both selected and background positions")
    return mask, aggregate


def fixed_roi_original_positions(torch: Any, token_layout: Mapping[str, Any], roi_mask: Any, device: Any) -> Any:
    """Keep L0/action and the same spatial ROI in every future frame."""

    spatial = len(token_layout["latent_positions"]["L1"])
    if tuple(roi_mask.shape) != (spatial,):
        raise ValueError(f"Expected ROI [{spatial}], got {tuple(roi_mask.shape)}")
    selected_spatial = torch.nonzero(roi_mask.to(device=device), as_tuple=False).flatten()
    parts = [
        torch.tensor(token_layout["latent_positions"]["L0"], dtype=torch.long, device=device),
    ]
    for latent in range(1, 9):
        frame = torch.tensor(token_layout["latent_positions"][f"L{latent}"], dtype=torch.long, device=device)
        parts.append(frame.index_select(0, selected_spatial))
    parts.append(torch.tensor(token_layout["action_positions"], dtype=torch.long, device=device))
    return torch.cat(parts).sort().values


def expand_token_roi_to_velocity_grid(
    *, torch: Any, roi_mask: Any, token_grid: tuple[int, int], velocity_grid: tuple[int, int]
) -> Any:
    """Nearest-expand a patch-token ROI and crop it to the velocity latent grid."""

    token_h, token_w = map(int, token_grid)
    velocity_h, velocity_w = map(int, velocity_grid)
    if tuple(roi_mask.shape) != (token_h * token_w,):
        raise ValueError(f"Expected flat ROI [{token_h * token_w}], got {tuple(roi_mask.shape)}")
    scale_h = (velocity_h + token_h - 1) // token_h
    scale_w = (velocity_w + token_w - 1) // token_w
    expanded = roi_mask.reshape(token_h, token_w).repeat_interleave(scale_h, 0).repeat_interleave(scale_w, 1)
    if int(expanded.shape[0]) < velocity_h or int(expanded.shape[1]) < velocity_w:
        raise RuntimeError("Token ROI cannot cover the velocity grid")
    return expanded[:velocity_h, :velocity_w].contiguous()


class Step0FixedROISparseController(ActionAttentionMass90Controller):
    """Dense step-0 profiler followed by a continuous B4..B27 sparse stack."""

    def __init__(
        self,
        *,
        torch: Any,
        net: Any,
        guidance: float,
        num_steps: int,
        threshold: float = 0.9,
        first_sparse_block: int = 4,
        output_dir: Path | None = None,
    ) -> None:
        super().__init__(
            torch=torch,
            net=net,
            guidance=guidance,
            num_steps=num_steps,
            threshold=threshold,
            output_dir=output_dir,
        )
        if not 0 < first_sparse_block < len(self.layers):
            raise ValueError("first_sparse_block must leave dense head and sparse tail blocks")
        self.first_sparse_block = int(first_sparse_block)
        self.fixed_roi_mask: Any | None = None
        self.fixed_roi_scores: Any | None = None
        self.vision_shape: tuple[int, ...] | None = None
        self._profile_tensors: list[Any] = []
        self._profile_rows: list[dict[str, Any]] = []
        self._profile_callback_block: int | None = None

    def _should_capture_step0_profile(self, block: int) -> bool:
        """Return whether a dense step-0 block contributes to ROI profiling."""

        return int(block) >= self.first_sparse_block

    def _network_post_hook(self, module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any], output: Any) -> None:
        if isinstance(output, dict) and output.get("preds_vision"):
            shape = tuple(int(value) for value in output["preds_vision"][0].shape)
            if self.vision_shape is None:
                self.vision_shape = shape
            elif shape != self.vision_shape:
                raise RuntimeError(f"Vision prediction shape changed: {self.vision_shape} -> {shape}")
        super()._network_post_hook(module, args, kwargs, output)

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
            raise RuntimeError("Step-0 profile callback has no live context")
        if layer_index != self._profile_callback_block:
            raise RuntimeError(f"Profile callback expected B{self._profile_callback_block}, got B{layer_index}")
        profiles = action_aligned_future_spatial_profiles(
            torch=self.torch,
            q_gen=q_gen,
            k_ar=k_ar,
            k_gen=k_gen,
            scaling=scaling,
            token_layout=self._layout,
        )
        self._profile_tensors.append(profiles.detach().cpu())
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
        if self.fixed_roi_mask is not None or self._current is None:
            return
        last_profile_branch = "conditional" if self.guidance == 1.0 else "unconditional"
        if int(self._current["step"]) != 0 or str(self._current["branch"]) != last_profile_branch:
            return
        expected = (1 if self.guidance == 1.0 else 2) * (len(self.layers) - self.first_sparse_block)
        if len(self._profile_tensors) != expected:
            raise RuntimeError(f"Expected {expected} step-0 block profiles, got {len(self._profile_tensors)}")
        stacked = self.torch.stack(self._profile_tensors)
        self.fixed_roi_mask, self.fixed_roi_scores = aggregate_fixed_roi(self.torch, stacked, self.threshold)

    def run_layer(
        self,
        *,
        block: int,
        decoder_layer: Any,
        hidden_states: SequencePack,
        attention_mask: Any,
        memory_value: Any,
        gen_only: bool,
    ) -> tuple[SequencePack, dict[str, Any], Any]:
        if not self.stack_active or self._current is None or self._layout is None:
            raise RuntimeError("run_layer called without an active fixed-ROI stack")
        if self._position_embeddings is None or self._active_original_positions is None or self._side_buffer is None:
            raise RuntimeError("Fixed-ROI stack state is incomplete")
        step = int(self._current["step"])

        # Step 0 and B0..B3 of later steps remain exactly dense.
        if step == 0 or block < self.first_sparse_block:
            attention = decoder_layer.self_attn
            capture = step == 0 and self._should_capture_step0_profile(block)
            if capture:
                if attention._attention_stats_capture_callback is not None:
                    raise RuntimeError(f"B{block} attention callback is occupied")
                self._profile_callback_block = block
                attention._attention_stats_capture_callback = self._capture_step0_profile
            try:
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
            self._side_buffer = full_gen.detach().clone()
            self._block_rows.append(
                {
                    **self._current,
                    "block": block,
                    "mode": "step0_profile_dense" if step == 0 else "dense_head",
                    "gen_tokens_before": int(full_gen.shape[0]),
                    "gen_tokens_after": int(full_gen.shape[0]),
                    "saved_gen_vs_original": 0,
                    "gen_retained_ratio": 1.0,
                    "finite": bool(self.torch.isfinite(full_gen).all()),
                }
            )
            return output, lbl_metadata, kv_to_store

        if self.fixed_roi_mask is None:
            raise RuntimeError("Later denoise step started before the step-0 fixed ROI was finalized")

        # Shrink once at B4. B5..B27 receive the same continuous sparse sequence.
        if block == self.first_sparse_block:
            selected_original = fixed_roi_original_positions(
                self.torch,
                self._layout,
                self.fixed_roi_mask,
                get_gen_seq(hidden_states).device,
            )
            selected_local = selected_original
            sparse_input, sparse_rope = self._slice_pack_and_rope(hidden_states, selected_local)
            hidden_states = sparse_input
            self._position_embeddings = sparse_rope
            self._active_original_positions = selected_original

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
            raise RuntimeError("Sparse block output length changed")
        self._side_buffer.index_copy_(0, self._active_original_positions, sparse_gen)
        original_gen = int(self._layout["num_gen_tokens"])
        retained = int(sparse_gen.shape[0])
        self._block_rows.append(
            {
                **self._current,
                "block": block,
                "mode": "fixed_roi_sparse",
                "gen_tokens_before": retained,
                "gen_tokens_after": retained,
                "saved_gen_vs_original": original_gen - retained,
                "gen_retained_ratio": retained / original_gen,
                "finite": bool(self.torch.isfinite(sparse_gen).all()),
            }
        )
        return output, lbl_metadata, kv_to_store

    def end_stack(self, hidden_states: SequencePack) -> SequencePack:
        if not self.stack_active or self._original_pack is None:
            raise RuntimeError("end_stack called without an active fixed-ROI stack")
        assert self._active_original_positions is not None and self._side_buffer is not None
        restored_gen = restore_terminal_hidden(
            torch=self.torch,
            side_buffer=self._side_buffer,
            active_hidden=get_gen_seq(hidden_states),
            active_positions=self._active_original_positions,
        )
        restored = from_und_gen_splits(get_und_seq(hidden_states), restored_gen, self._original_pack)
        self._completed_stacks += 1
        self._finalize_roi_if_ready()
        self.abort_stack()
        return restored

    def finish(self) -> dict[str, Any]:
        expected_stacks = self.num_steps if self.guidance == 1.0 else 2 * self.num_steps
        expected_blocks = expected_stacks * len(self.layers)
        if self.stack_active or self.fixed_roi_mask is None or self.fixed_roi_scores is None:
            raise RuntimeError("Fixed-ROI experiment did not complete")
        if self._completed_stacks != expected_stacks or len(self._block_rows) != expected_blocks:
            raise RuntimeError(
                f"Incomplete stacks={self._completed_stacks}/{expected_stacks}, "
                f"blocks={len(self._block_rows)}/{expected_blocks}"
            )
        if not all(bool(row["finite"]) for row in self._block_rows):
            raise RuntimeError("Sparse output contains NaN/Inf")
        assert self._layout is not None
        spatial = int(self.fixed_roi_mask.numel())
        roi = int(self.fixed_roi_mask.sum())
        full_gen = int(self._layout["num_gen_tokens"])
        sparse_gen = full_gen - 8 * (spatial - roi)
        sparse_calls = (self.num_steps - 1) * (1 if self.guidance == 1.0 else 2) * (
            len(self.layers) - self.first_sparse_block
        )
        total_calls = expected_blocks
        summary = {
            "schema_version": 1,
            "strategy_version": VELOCITY_CACHE_STRATEGY_VERSION,
            "experiment": "step0_profile_fixed_roi_guided_background_velocity_cache",
            "threshold": self.threshold,
            "aggregation": "max over step0 branch x B4-B27 x L1-L8, then minimum mass mask",
            "first_sparse_block": self.first_sparse_block,
            "step0_all_blocks_dense": True,
            "later_dense_blocks": list(range(self.first_sparse_block)),
            "later_sparse_blocks": list(range(self.first_sparse_block, len(self.layers))),
            "future_frames_share_spatial_mask": True,
            "cfg_branches_share_spatial_mask": True,
            "roi_tokens_per_future": roi,
            "background_tokens_per_future": spatial - roi,
            "selected_spatial_positions": self.torch.nonzero(
                self.fixed_roi_mask, as_tuple=False
            ).flatten().tolist(),
            "full_gen_tokens": full_gen,
            "sparse_gen_tokens": sparse_gen,
            "sparse_gen_retained_ratio": sparse_gen / full_gen,
            "saved_gen_tokens_per_sparse_block": full_gen - sparse_gen,
            "sparse_block_calls": sparse_calls,
            "total_block_calls": total_calls,
            "average_saved_gen_tokens_per_block_call": sparse_calls * (full_gen - sparse_gen) / total_calls,
            "profile_count": len(self._profile_tensors),
            "vision_shape": list(self.vision_shape or ()),
            "all_finite": True,
        }
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.torch.save(
                {
                    "profiles": self.torch.stack(self._profile_tensors),
                    "aggregate_scores": self.fixed_roi_scores,
                    "fixed_roi_mask": self.fixed_roi_mask,
                },
                self.output_dir / "step0_roi_profile.pt",
            )
            with (self.output_dir / "step0_profile_rows.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self._profile_rows[0]))
                writer.writeheader()
                writer.writerows(self._profile_rows)
            with (self.output_dir / "block_token_savings.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self._block_rows[0]))
                writer.writeheader()
                writer.writerows(self._block_rows)
            (self.output_dir / "token_savings_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        return summary


def _flat_metrics(torch: Any, reference: Any, candidate: Any, eps: float = 1e-12) -> dict[str, float]:
    ref = reference.detach().float().reshape(-1)
    got = candidate.detach().float().reshape(-1)
    diff = got - ref
    return {
        "cosine": float(torch.nn.functional.cosine_similarity(ref, got, dim=0, eps=eps)),
        "relative_l2": float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(ref).clamp_min(eps)),
        "mse": float(diff.square().mean()),
        "max_absolute_error": float(diff.abs().max()),
    }


def merge_cached_background_velocity(
    *, torch: Any, current: Any, cached_step0: Any, roi_mask: Any
) -> tuple[Any, dict[str, float]]:
    """Use current ROI and step-0 background in ``[C,T,H,W]`` or ``[B,C,T,H,W]``."""

    if current.shape != cached_step0.shape or current.ndim not in (4, 5):
        raise ValueError(
            "Current/cached vision velocities must be matching [C,T,H,W] or [B,C,T,H,W], "
            f"got {tuple(current.shape)} and {tuple(cached_step0.shape)}"
        )
    temporal_dim = current.ndim - 3
    if int(current.shape[temporal_dim]) != 9 or tuple(roi_mask.shape) != tuple(current.shape[-2:]):
        raise ValueError("Expected L0+L1..L8 and an HxW ROI mask")
    current_future = current.narrow(temporal_dim, 1, 8)
    cached_future = cached_step0.narrow(temporal_dim, 1, 8).to(device=current.device, dtype=current.dtype)
    view_shape = [1] * (current_future.ndim - 2) + [int(current.shape[-2]), int(current.shape[-1])]
    background = (~roi_mask.to(device=current.device)).reshape(view_shape).expand_as(current_future)
    merged = current.clone()
    merged.narrow(temporal_dim, 1, 8).copy_(torch.where(background, cached_future, current_future))
    metrics = _flat_metrics(torch, current_future[background], cached_future[background])
    metrics["background_elements"] = int(background.sum())
    if not bool(torch.isfinite(merged).all()):
        raise RuntimeError("Merged guided velocity contains NaN/Inf")
    return merged, metrics


class GuidedVelocityTraceSampler:
    """Transparent sampler wrapper that records guided velocity per denoise step."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.velocities: list[list[Any]] = []
        self.timesteps: list[float] = []

    def __call__(self, velocity_fn: Any, initial_noise: Any, **kwargs: Any) -> Any:
        self.velocities.clear()
        self.timesteps.clear()

        def traced(noise_x: Any, timestep: Any) -> Any:
            velocity = velocity_fn(noise_x, timestep)
            self.velocities.append([item.detach().clone() for item in velocity])
            self.timesteps.append(float(timestep.reshape(-1)[0]))
            return velocity

        return self.inner(traced, initial_noise, **kwargs)


class GuidedBackgroundVelocityCacheSampler(GuidedVelocityTraceSampler):
    """Merge cached step-0 background into later guided velocities before UniPC."""

    def __init__(
        self,
        inner: Any,
        controller: Step0FixedROISparseController,
        reference_velocities: Sequence[Sequence[Any]] | None = None,
    ) -> None:
        super().__init__(inner)
        self.controller = controller
        self.reference_velocities = reference_velocities
        self.cache_trace: list[dict[str, Any]] = []
        self.cached_step0_vision: list[Any] = []

    def __call__(self, velocity_fn: Any, initial_noise: Any, **kwargs: Any) -> Any:
        self.velocities.clear()
        self.timesteps.clear()
        self.cache_trace.clear()
        self.cached_step0_vision.clear()
        step = 0

        def cached(noise_x: Any, timestep: Any) -> Any:
            nonlocal step
            current = velocity_fn(noise_x, timestep)
            if self.controller.vision_shape is None or self.controller.fixed_roi_mask is None:
                raise RuntimeError("Controller did not expose vision shape/fixed ROI after guided velocity")
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
                        raise RuntimeError("Controller token layout is unavailable")
                    _, token_h, token_w = self.controller._layout["latent_shape_thw"]
                    velocity_roi = expand_token_roi_to_velocity_grid(
                        torch=self.controller.torch,
                        roi_mask=self.controller.fixed_roi_mask,
                        token_grid=(int(token_h), int(token_w)),
                        velocity_grid=(int(vision.shape[-2]), int(vision.shape[-1])),
                    )
                    merged, cache_metrics = merge_cached_background_velocity(
                        torch=self.controller.torch,
                        current=vision,
                        cached_step0=self.cached_step0_vision[sample],
                        roi_mask=velocity_roi,
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
                raise RuntimeError("Cached velocity output contains NaN/Inf")
            self.velocities.append([item.detach().clone() for item in output])
            self.timesteps.append(float(timestep.reshape(-1)[0]))
            self.cache_trace.append(step_record)
            step += 1
            return output

        result = self.inner(cached, initial_noise, **kwargs)
        expected = int(kwargs.get("num_steps", self.controller.num_steps))
        if step != expected:
            raise RuntimeError(f"Expected {expected} guided velocity calls, got {step}")
        return result

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "velocity_cache_trace.json").write_text(
            json.dumps(self.cache_trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
