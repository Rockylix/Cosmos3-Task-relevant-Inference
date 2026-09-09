# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Core-then-Stable sparse denoising for Cosmos3-Edge RoboLab inference.

The conditional branch of denoising step 0 runs densely and profiles all 28
decoder blocks.  Its action-to-future attention selects a per-frame Core-80
and a cross-frame Stable-104 mask.  The resulting K184 mask is reused by every
decoder block in all remaining CFG passes and denoising steps.  L0 and action
tokens always remain live Q/K/V tokens.

The policy uses the native sampler. Each stack restores unselected future
positions from that stack's input before the sampler consumes the output.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    from_und_gen_splits,
    get_gen_seq,
    get_und_seq,
    init_sequence_pack,
)
from cosmos_framework.inference.edge_core_stable_layout import _action_query_layout
from cosmos_framework.model.attention import attention

STRATEGY_VERSION = "edge-core80-stable104-action-weighted"
NUM_DENOISE_STEPS = 4
PROFILE_BLOCKS = tuple(range(28))
TOKEN_BUDGET = 184
CORE_TOKEN_BUDGET = 80
STABLE_TOKEN_BUDGET = TOKEN_BUDGET - CORE_TOKEN_BUDGET
CORE_BLOCK_COUNT = 6
STABLE_CV_PENALTY = 1.0
ACTION_HORIZON_WEIGHTS = (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0)


def _make_sequence_pack(*, und_seq: Any, gen_seq: Any) -> SequencePack:
    und_len = int(und_seq.shape[0])
    gen_len = int(gen_seq.shape[0])
    metadata = init_sequence_pack(
        sample_lens=[und_len + gen_len],
        split_lens=[und_len, gen_len],
        attn_modes=["causal", "full"],
        device=gen_seq.device,
    )
    return {
        **metadata,
        "max_num_tokens": und_len + gen_len,
        "causal_seq": und_seq,
        "full_only_seq": gen_seq,
        "is_sharded": False,
    }


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
        raise ValueError(f"Cannot select {count} tokens from {int(candidates.numel())} candidates")
    chosen = torch.topk(scores.index_select(0, candidates), count, sorted=False).indices
    mask[candidates.index_select(0, chosen)] = True
    return mask


def _validate_profile_inputs(
    *,
    torch: Any,
    q_gen: Any,
    k_ar: Any,
    k_gen: Any,
    token_layout: Mapping[str, Any],
) -> tuple[Any, Any]:
    if q_gen.ndim != 3 or k_ar.ndim != 3 or k_gen.ndim != 3:
        raise ValueError("Expected Q/K tensors shaped [tokens,heads,head_dim]")
    if int(q_gen.shape[0]) != int(token_layout["num_gen_tokens"]):
        raise RuntimeError("Version1 profiling requires the complete GEN sequence")
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
    weights = torch.tensor(ACTION_HORIZON_WEIGHTS, dtype=torch.float32, device=q_gen.device)
    return action_index, weights


def action_aligned_future_profiles_with_lse(
    *,
    torch: Any,
    q_gen: Any,
    k_ar: Any,
    k_gen: Any,
    v_ar: Any,
    v_gen: Any,
    scaling: float,
    token_layout: Mapping[str, Any],
) -> Any:
    """Compute the same profile using the attention kernel's online LSE."""

    if k_ar.shape != v_ar.shape or k_gen.shape != v_gen.shape:
        raise RuntimeError("K/V geometry differs during Version1 profiling")
    action_index, weights = _validate_profile_inputs(
        torch=torch,
        q_gen=q_gen,
        k_ar=k_ar,
        k_gen=k_gen,
        token_layout=token_layout,
    )
    q_heads = int(q_gen.shape[1])
    kv_heads = int(k_gen.shape[1])
    q_action = q_gen.index_select(0, action_index)
    k_all = torch.cat((k_ar, k_gen), dim=0)
    v_all = torch.cat((v_ar, v_gen), dim=0)
    _output, lse = attention(
        query=q_action.unsqueeze(0),
        key=k_all.unsqueeze(0),
        value=v_all.unsqueeze(0),
        scale=float(scaling),
        return_lse=True,
    )
    del _output
    lse = lse.squeeze(0)
    if lse.ndim == 3 and int(lse.shape[-1]) == 1:
        lse = lse.squeeze(-1)
    if tuple(lse.shape) != (32, q_heads):
        raise RuntimeError(f"Unexpected attention LSE shape {tuple(lse.shape)}")

    lse_hq = lse.detach().float().transpose(0, 1)
    q_action_hqd = q_action.detach().float().permute(1, 0, 2).contiguous()
    profiles = []
    for latent in range(1, 9):
        horizons = list(range(4 * (latent - 1), 4 * latent))
        positions = torch.tensor(
            token_layout["latent_positions"][f"L{latent}"],
            dtype=torch.long,
            device=q_gen.device,
        )
        k_frame = k_gen.index_select(0, positions).detach().float()
        k_frame = k_frame.repeat_interleave(q_heads // kv_heads, dim=1).permute(1, 0, 2).contiguous()
        logits = torch.einsum(
            "hqd,hkd->hqk",
            q_action_hqd[:, horizons] * float(scaling),
            k_frame,
        )
        probabilities = torch.exp(logits - lse_hq[:, horizons, None]).mean(dim=0)
        profiles.append((probabilities * weights[:, None]).sum(dim=0))
    result = torch.stack(profiles)
    if not bool(torch.isfinite(result).all()) or bool((result < 0).any()):
        raise RuntimeError("Version1 fused Action-Relevance profile is invalid")
    return result


def build_core_stable_plan(
    *,
    torch: Any,
    profile_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select Core first, then Stable, and return one fixed K184 mask."""

    lookup = {int(record["block"]): record["profiles"] for record in profile_records}
    if sorted(lookup) != list(PROFILE_BLOCKS) or len(lookup) != len(profile_records):
        raise RuntimeError("Version1 requires exactly one conditional profile for every block B0..B27")
    raw = torch.stack([lookup[block] for block in PROFILE_BLOCKS])
    if raw.ndim != 3 or int(raw.shape[1]) != 8:
        raise ValueError(f"Expected profiles [28,8,spatial], got {tuple(raw.shape)}")
    if not bool(torch.isfinite(raw).all()) or bool((raw < 0).any()):
        raise RuntimeError("Version1 profile tensor contains NaN/Inf or negative values")

    blocks, frames, spatial = map(int, raw.shape)
    if not 0 < CORE_BLOCK_COUNT <= blocks:
        raise ValueError("Core block count does not fit the profiled decoder")
    if CORE_TOKEN_BUDGET + STABLE_TOKEN_BUDGET > spatial:
        raise ValueError("Core and Stable budgets exceed the spatial token grid")

    eps = torch.finfo(raw.dtype).eps
    frame_mass = raw.sum(dim=-1)
    normalized = raw / frame_mass.unsqueeze(-1).clamp_min(eps)
    entropy = -(normalized * normalized.clamp_min(eps).log()).sum(dim=-1) / math.log(spatial)
    entropy = torch.where(frame_mass > eps, entropy, torch.ones_like(entropy))
    block_mass = frame_mass.mean(dim=1)
    block_entropy = entropy.mean(dim=1)
    block_quality = block_mass * (1.0 - block_entropy).clamp_min(0)

    core_block_indices = torch.topk(block_quality, CORE_BLOCK_COUNT).indices
    core_quality = block_quality.index_select(0, core_block_indices)
    core_weights = core_quality / core_quality.sum().clamp_min(eps)
    core_scores = (raw.index_select(0, core_block_indices) * core_weights[:, None, None]).sum(dim=0)

    global_weights = block_quality
    if float(global_weights.sum()) <= float(eps):
        global_weights = torch.ones_like(global_weights)
    global_weights = global_weights / global_weights.sum()
    global_scores = (raw * global_weights[:, None, None]).sum(dim=0)
    stable_mean = global_scores.mean(dim=0)
    stable_std = global_scores.std(dim=0, unbiased=False)
    stable_cv = stable_std / stable_mean.clamp_min(eps)
    stable_scores = stable_mean / (1.0 + float(STABLE_CV_PENALTY) * stable_cv)

    # Reserve enough global coordinates for Stable before selecting frame-local
    # Core tokens.  Core remains the first selected component, and Stable is
    # subsequently ranked only outside the resulting cross-frame Core union.
    core_pool_budget = spatial - STABLE_TOKEN_BUDGET
    core_pool_mask = _select_mask(torch, core_scores.amax(dim=0), core_pool_budget)
    core_masks = torch.stack(
        [_select_mask(torch, core_scores[frame], CORE_TOKEN_BUDGET, core_pool_mask) for frame in range(frames)]
    )
    core_union = core_masks.any(dim=0)
    stable_mask = _select_mask(torch, stable_scores, STABLE_TOKEN_BUDGET, ~core_union)
    execution_mask = core_masks | stable_mask.unsqueeze(0)
    if execution_mask.sum(dim=-1).tolist() != [CORE_TOKEN_BUDGET + STABLE_TOKEN_BUDGET] * frames:
        raise RuntimeError("Version1 failed exact K184 validation")
    if bool((core_union & stable_mask).any()):
        raise RuntimeError("Version1 Stable mask overlaps the Core union")

    return {
        "raw_profiles": raw,
        "block_mass": block_mass,
        "block_entropy": block_entropy,
        "block_quality": block_quality,
        "core_blocks": [int(PROFILE_BLOCKS[int(index)]) for index in core_block_indices],
        "core_weights": core_weights,
        "core_scores": core_scores,
        "core_pool_mask": core_pool_mask,
        "core_masks": core_masks,
        "stable_mean": stable_mean,
        "stable_std": stable_std,
        "stable_cv": stable_cv,
        "stable_scores": stable_scores,
        "stable_mask": stable_mask,
        "execution_mask": execution_mask,
    }


def _selected_original_positions(torch: Any, token_layout: Mapping[str, Any], mask: Any, device: Any) -> Any:
    spatial = len(token_layout["latent_positions"]["L1"])
    if tuple(mask.shape) != (8, spatial):
        raise ValueError(f"Expected mask [8,{spatial}], got {tuple(mask.shape)}")
    parts = [torch.tensor(token_layout["latent_positions"]["L0"], dtype=torch.long, device=device)]
    for latent in range(1, 9):
        frame = torch.tensor(token_layout["latent_positions"][f"L{latent}"], dtype=torch.long, device=device)
        selected = torch.nonzero(mask[latent - 1].to(device=device), as_tuple=False).flatten()
        parts.append(frame.index_select(0, selected))
    parts.append(torch.tensor(token_layout["action_positions"], dtype=torch.long, device=device))
    return torch.cat(parts).sort().values


class Version1Controller:
    """Request-local controller for the fixed C80/S104 policy."""

    def __init__(
        self,
        *,
        torch: Any,
        net: Any,
        guidance: float,
        num_steps: int,
        output_dir: Path | None = None,
    ) -> None:
        if int(num_steps) != NUM_DENOISE_STEPS or float(guidance) != 3.0:
            raise ValueError("The fixed Edge policy requires CFG 3 and 4 denoising steps")
        self.torch = torch
        self.net = net
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.model = net.language_model.model
        self.layers = list(self.model.layers)
        if len(self.layers) != len(PROFILE_BLOCKS):
            raise ValueError(f"Version1 expects 28 decoder blocks, got {len(self.layers)}")

        self.stack_active = False
        self._handles: list[Any] = []
        self._current: dict[str, Any] | None = None
        self._layout: dict[str, Any] | None = None
        self._forward_call_index = 0
        self._original_pack: SequencePack | None = None
        self._position_embeddings: tuple[SequencePack, SequencePack] | None = None
        self._active_original_positions: Any | None = None
        self._side_buffer: Any | None = None
        self._selected_positions: Any | None = None
        self._selected_positions_device: Any | None = None
        self._profile_callback_block: int | None = None
        self._profile_records: list[dict[str, Any]] = []
        self.plan: dict[str, Any] | None = None
        self._completed_stacks = 0
        self._dense_stacks = 0
        self._sparse_stacks = 0
        self._sparse_block_calls = 0

    def __enter__(self) -> "Version1Controller":
        if hasattr(self.model, "_robolab_version1_controller"):
            raise RuntimeError("A RoboLab Version1 controller is already installed")
        for layer in self.layers:
            if layer.self_attn._attention_stats_capture_callback is not None:
                raise RuntimeError("Version1 conflicts with an active attention capture callback")
        setattr(self.model, "_robolab_version1_controller", self)
        self._handles.append(self.net.register_forward_pre_hook(self._network_pre_hook, with_kwargs=True))
        self._handles.append(
            self.net.register_forward_hook(self._network_post_hook, with_kwargs=True, always_call=True)
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.abort_stack()
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        if getattr(self.model, "_robolab_version1_controller", None) is self:
            delattr(self.model, "_robolab_version1_controller")
        self._current = None

    def _call_semantics(self, call_index: int) -> tuple[int, str]:
        return call_index // 2, "conditional" if call_index % 2 == 0 else "unconditional"

    def _network_pre_hook(self, module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
        del module
        und_only = bool(kwargs.get("und_only", args[2] if len(args) > 2 else False))
        if und_only:
            self._current = None
            return
        packed_sequence = args[0] if args else kwargs.get("packed_seq")
        step, branch = self._call_semantics(self._forward_call_index)
        self._forward_call_index += 1
        layout = _action_query_layout(self.torch, packed_sequence)
        if self._layout is None:
            self._layout = layout
        elif layout != self._layout:
            raise RuntimeError("GEN/action token layout changed during Version1 inference")
        self._current = {"step": step, "branch": branch}

    def _network_post_hook(self, module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any], output: Any) -> None:
        del module, args, kwargs, output
        if self.stack_active:
            self.abort_stack()
            raise RuntimeError("Network forward ended before Version1 restored the full sequence")
        self._current = None

    def begin_stack(
        self,
        *,
        hidden_states: SequencePack,
        position_embeddings: tuple[SequencePack, SequencePack],
        natten_metadata_list: list | None,
    ) -> None:
        if self._current is None:
            return
        if self.stack_active:
            raise RuntimeError("Nested Version1 Transformer stacks are unsupported")
        if natten_metadata_list is not None:
            raise RuntimeError("Version1 requires dense two-way attention rather than NATTEN")
        if self._layout is None:
            raise RuntimeError("Version1 token layout is unavailable")
        num_gen = int(hidden_states["_num_full_tokens"])
        if num_gen != int(self._layout["num_gen_tokens"]):
            raise RuntimeError("Version1 input GEN length differs from its token layout")
        if bool(hidden_states.get("is_sharded", False)):
            raise RuntimeError("Version1 does not support context-parallel SequencePacks")
        self._original_pack = hidden_states
        self._position_embeddings = position_embeddings
        self._active_original_positions = None
        self._side_buffer = (
            None
            if self._current == {"step": 0, "branch": "conditional"}
            else get_gen_seq(hidden_states).detach().clone()
        )
        self.stack_active = True

    def _capture_profile(self, **kwargs: Any) -> None:
        block = int(kwargs["layer_index"])
        if self._profile_callback_block != block or self._layout is None:
            raise RuntimeError("Version1 attention callback has stale block state")
        profile = action_aligned_future_profiles_with_lse(
            torch=self.torch,
            q_gen=kwargs["q_gen"],
            k_ar=kwargs["k_ar"],
            k_gen=kwargs["k_gen"],
            v_ar=kwargs["v_ar"],
            v_gen=kwargs["v_gen"],
            scaling=float(kwargs["scaling"]),
            token_layout=self._layout,
        ).detach()
        self._profile_records.append({"block": block, "profiles": profile})

    def _run_dense_profile_layer(
        self,
        *,
        block: int,
        decoder_layer: Any,
        hidden_states: SequencePack,
        attention_mask: Any,
        memory_value: Any,
        gen_only: bool,
    ) -> tuple[SequencePack, dict[str, Any], Any]:
        assert self._position_embeddings is not None
        attention_module = decoder_layer.self_attn
        if attention_module._attention_stats_capture_callback is not None:
            raise RuntimeError(f"B{block} attention callback is already occupied")
        self._profile_callback_block = block
        attention_module._attention_stats_capture_callback = self._capture_profile
        try:
            return decoder_layer(
                hidden_states,
                attention_mask,
                self._position_embeddings,
                natten_metadata=None,
                memory_value=memory_value,
                gen_only=gen_only,
            )
        finally:
            attention_module._attention_stats_capture_callback = None
            self._profile_callback_block = None

    def _prepare_selected_positions(self, device: Any) -> Any:
        if self._selected_positions is not None:
            if device != self._selected_positions_device:
                raise RuntimeError("Version1 GEN token device changed within a request")
            return self._selected_positions
        if self.plan is None or self._layout is None:
            raise RuntimeError("Version1 cannot pack tokens before mask selection")
        self._selected_positions = _selected_original_positions(
            self.torch,
            self._layout,
            self.plan["execution_mask"],
            device,
        )
        self._selected_positions_device = device
        return self._selected_positions

    def _slice_pack_and_rope(self, hidden_states: SequencePack, selected_local: Any) -> tuple[SequencePack, Any]:
        assert self._position_embeddings is not None
        sparse_pack = _make_sequence_pack(
            und_seq=get_und_seq(hidden_states),
            gen_seq=get_gen_seq(hidden_states).index_select(0, selected_local),
        )
        cos_pack, sin_pack = self._position_embeddings
        sparse_cos = _make_sequence_pack(
            und_seq=get_und_seq(cos_pack),
            gen_seq=get_gen_seq(cos_pack).index_select(0, selected_local),
        )
        sparse_sin = _make_sequence_pack(
            und_seq=get_und_seq(sin_pack),
            gen_seq=get_gen_seq(sin_pack).index_select(0, selected_local),
        )
        return sparse_pack, (sparse_cos, sparse_sin)

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
        if not self.stack_active or self._current is None:
            raise RuntimeError("Version1 run_layer called without an active stack")
        if self._position_embeddings is None:
            raise RuntimeError("Version1 sparse stack state is incomplete")
        dense_profile = int(self._current["step"]) == 0 and str(self._current["branch"]) == "conditional"
        if dense_profile:
            return self._run_dense_profile_layer(
                block=block,
                decoder_layer=decoder_layer,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                memory_value=memory_value,
                gen_only=gen_only,
            )
        if self.plan is None:
            raise RuntimeError("Sparse CFG pass started before Version1 selected its mask")

        if block == 0:
            selected_original = self._prepare_selected_positions(get_gen_seq(hidden_states).device)
            hidden_states, self._position_embeddings = self._slice_pack_and_rope(hidden_states, selected_original)
            self._active_original_positions = selected_original

        output, lbl_metadata, kv_to_store = decoder_layer(
            hidden_states,
            attention_mask,
            self._position_embeddings,
            natten_metadata=None,
            memory_value=memory_value,
            gen_only=gen_only,
        )
        if int(get_gen_seq(output).shape[0]) != int(self._active_original_positions.numel()):
            raise RuntimeError("Version1 sparse block changed the packed GEN length")
        self._sparse_block_calls += 1
        return output, lbl_metadata, kv_to_store

    def end_stack(self, hidden_states: SequencePack) -> SequencePack:
        if not self.stack_active or self._original_pack is None or self._current is None:
            raise RuntimeError("Version1 end_stack called without an active stack")
        dense_profile = int(self._current["step"]) == 0 and str(self._current["branch"]) == "conditional"
        if dense_profile:
            self.plan = build_core_stable_plan(
                torch=self.torch,
                profile_records=self._profile_records,
            )
            restored = hidden_states
            self._dense_stacks += 1
        else:
            assert self._active_original_positions is not None and self._side_buffer is not None
            restored_gen = self._side_buffer.clone()
            restored_gen.index_copy_(0, self._active_original_positions, get_gen_seq(hidden_states))
            restored = from_und_gen_splits(get_und_seq(hidden_states), restored_gen, self._original_pack)
            self._sparse_stacks += 1
        self._completed_stacks += 1
        self.abort_stack()
        return restored

    def abort_stack(self) -> None:
        self.stack_active = False
        self._original_pack = None
        self._position_embeddings = None
        self._active_original_positions = None
        self._side_buffer = None
        self._profile_callback_block = None

    def finish(self) -> dict[str, Any]:
        expected_stacks = NUM_DENOISE_STEPS * 2
        expected_dense = 1
        expected_sparse = expected_stacks - expected_dense
        if self.stack_active or self.plan is None:
            raise RuntimeError("Version1 request did not finish cleanly")
        if (self._completed_stacks, self._dense_stacks, self._sparse_stacks) != (
            expected_stacks,
            expected_dense,
            expected_sparse,
        ):
            raise RuntimeError(
                "Unexpected Version1 stack counts: "
                f"total={self._completed_stacks}/{expected_stacks}, "
                f"dense={self._dense_stacks}/{expected_dense}, sparse={self._sparse_stacks}/{expected_sparse}"
            )
        if self._sparse_block_calls != expected_sparse * len(self.layers):
            raise RuntimeError("Version1 did not sparsify every expected decoder block")

        summary = {
            "schema_version": 2,
            "strategy_version": STRATEGY_VERSION,
            "selection_order": "core_then_stable",
            "profile": "conditional_step0_B0_B27",
            "core_blocks": self.plan["core_blocks"],
            "core_block_count": CORE_BLOCK_COUNT,
            "core_token_budget": CORE_TOKEN_BUDGET,
            "stable_token_budget": STABLE_TOKEN_BUDGET,
            "token_budget": TOKEN_BUDGET,
            "action_horizon_weights": list(ACTION_HORIZON_WEIGHTS),
            "stable_cv_penalty": STABLE_CV_PENALTY,
            "sampler": "baseline_unipc_unmodified",
            "unselected_future_hidden_restore": "current_stack_input",
            "dense_stack_count": self._dense_stacks,
            "sparse_stack_count": self._sparse_stacks,
        }
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            artifact = {
                "strategy_version": STRATEGY_VERSION,
                "core_blocks": self.plan["core_blocks"],
                "block_quality": self.plan["block_quality"].detach().cpu(),
                "core_masks": self.plan["core_masks"].detach().cpu(),
                "stable_mask": self.plan["stable_mask"].detach().cpu(),
                "execution_mask": self.plan["execution_mask"].detach().cpu(),
            }
            self.torch.save(artifact, self.output_dir / "selection.pt")
            (self.output_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return summary
