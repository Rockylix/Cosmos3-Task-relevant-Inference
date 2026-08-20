# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Oracle Action-attention-guided future-token intervention for RoboLab.

This is deliberately an eager-only correctness experiment, not an acceleration
feature.  Every decoder block first runs a read-only probe on the currently
active sequence.  The probe's exact post-RoPE Q/K defines a 90%-mass union of
future spatial tokens.  The same block input is then re-run on that shorter
sequence and only the second result is committed.

Dropped future tokens never re-enter later decoder blocks.  A side buffer keeps
their last valid hidden value solely to restore the fixed full-grid contract
before the final RMSNorm and modality output heads.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    from_und_gen_splits,
    get_gen_seq,
    get_und_seq,
    init_sequence_pack,
)
from cosmos_framework.scripts.robolab_action_query_attention_capture import _action_query_layout

TOKEN_FIELDS = [
    "step",
    "timestep",
    "branch",
    "block",
    "latent",
    "active_before",
    "selected_after",
    "dropped_this_block",
    "saved_vs_original",
    "retained_vs_original",
    "selected_spatial_positions",
    "coverage_min",
    "coverage_mean",
]

BLOCK_FIELDS = [
    "step",
    "timestep",
    "branch",
    "block",
    "ar_tokens",
    "gen_tokens_before",
    "gen_tokens_after",
    "future_tokens_before",
    "future_tokens_after",
    "shared_future_spatial_tokens",
    "dropped_this_block",
    "saved_gen_vs_original",
    "saved_future_vs_original",
    "gen_retained_ratio",
    "total_b27_style_key_retained_ratio",
    "attention_work_ratio_vs_current_probe",
    "finite",
]


def make_sequence_pack(*, und_seq: Any, gen_seq: Any) -> SequencePack:
    """Build an unpadded, single-sample two-way pack for an irregular GEN set."""

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


def minimum_mass_mask(weights: Any, threshold: float) -> Any:
    """Return per-row minimal top-weight masks reaching ``threshold`` mass.

    ``weights`` is ``[queries,tokens]`` and need not be normalized.  At least
    one token is selected for every non-empty row.
    """

    if weights.ndim != 2 or int(weights.shape[1]) == 0:
        raise ValueError(f"Expected non-empty [queries,tokens] weights, got {tuple(weights.shape)}")
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0,1], got {threshold}")
    sorted_weights, order = weights.sort(dim=-1, descending=True)
    cumulative = sorted_weights.cumsum(dim=-1)
    target = weights.sum(dim=-1, keepdim=True) * float(threshold)
    counts = (cumulative < target).sum(dim=-1).add(1).clamp_max(int(weights.shape[1]))
    ranks = order.new_empty(order.shape)
    rank_values = order.new_tensor(list(range(int(weights.shape[1])))).expand_as(order)
    ranks.scatter_(dim=-1, index=order, src=rank_values)
    return ranks < counts.unsqueeze(-1)


def select_future_mass_union(
    *,
    torch: Any,
    q_gen: Any,
    k_ar: Any,
    k_gen: Any,
    scaling: float,
    token_layout: Mapping[str, Any],
    active_original_positions: Any,
    threshold: float,
    eps: float = 1e-12,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    """Select one shared spatial mask covering every query/frame 90%-mass set.

    L1..L8 must enter the committed sparse block with exactly the same spatial
    coordinates.  We therefore compute the minimal set for every
    ``(predicted_action_query, future_frame)`` independently, take one union in
    spatial-coordinate space, and apply that shared union to all eight frames.
    """

    num_active = int(active_original_positions.numel())
    if int(q_gen.shape[0]) != num_active or int(k_gen.shape[0]) != num_active:
        raise RuntimeError("Live Q/K length differs from active GEN position count")
    if q_gen.ndim != 3 or k_ar.ndim != 3 or k_gen.ndim != 3:
        raise RuntimeError("Expected Q/K tensors in [tokens,heads,head_dim]")
    num_q_heads = int(q_gen.shape[1])
    num_kv_heads = int(k_gen.shape[1])
    if num_q_heads % num_kv_heads:
        raise RuntimeError(f"Invalid GQA geometry: q_heads={num_q_heads}, kv_heads={num_kv_heads}")

    original_to_local = {int(value): index for index, value in enumerate(active_original_positions.tolist())}
    predicted_positions = [
        int(query["gen_position"]) for query in token_layout["action_queries"] if query["query_role"] == "predicted"
    ]
    if len(predicted_positions) != 32:
        raise RuntimeError(f"Expected 32 predicted Action Queries, got {len(predicted_positions)}")
    try:
        action_local = torch.tensor(
            [original_to_local[position] for position in predicted_positions],
            dtype=torch.long,
            device=q_gen.device,
        )
    except KeyError as exc:
        raise RuntimeError(f"Required Action Query token {int(exc.args[0])} was pruned") from exc

    q_heads = q_gen.index_select(0, action_local).detach().float().permute(1, 0, 2).contiguous()
    k_all = torch.cat((k_ar, k_gen), dim=0).detach().float()
    gqa_repeat = num_q_heads // num_kv_heads
    k_heads = k_all.repeat_interleave(gqa_repeat, dim=1).permute(1, 0, 2).contiguous()
    probabilities = torch.softmax(torch.matmul(q_heads, k_heads.transpose(1, 2)) * float(scaling), dim=-1)
    head_mean = probabilities.mean(dim=0)  # [predicted_action_query,key]
    if not bool(torch.isfinite(head_mean).all()):
        raise RuntimeError("Action-to-key attention probabilities contain NaN/Inf")

    keep_local = torch.zeros(num_active, dtype=torch.bool, device=q_gen.device)
    future_original = {
        int(position) for latent in range(1, 9) for position in token_layout["latent_positions"][f"L{latent}"]
    }
    # L0, q0 and q1..q32 (and any future layout extension) are mandatory.
    for local, original in enumerate(active_original_positions.tolist()):
        if int(original) not in future_original:
            keep_local[local] = True

    frame_inputs: list[dict[str, Any]] = []
    num_ar = int(k_ar.shape[0])
    shared_active_spatial: list[int] | None = None
    shared_union_mask: Any | None = None
    for latent in range(1, 9):
        original_frame = list(map(int, token_layout["latent_positions"][f"L{latent}"]))
        active_spatial = [
            spatial_index for spatial_index, position in enumerate(original_frame) if position in original_to_local
        ]
        active_frame_original = [original_frame[spatial_index] for spatial_index in active_spatial]
        if not active_frame_original:
            raise RuntimeError(f"All tokens from L{latent} were pruned")
        if shared_active_spatial is None:
            shared_active_spatial = active_spatial
        elif active_spatial != shared_active_spatial:
            raise RuntimeError(
                "Future frames do not share the same active spatial coordinates "
                f"before L{latent}: expected={shared_active_spatial}, got={active_spatial}"
            )
        frame_local = torch.tensor(
            [original_to_local[position] for position in active_frame_original],
            dtype=torch.long,
            device=q_gen.device,
        )
        key_index = frame_local + num_ar
        frame_weights = head_mean.index_select(-1, key_index)
        per_query_mask = minimum_mass_mask(frame_weights, threshold)
        frame_union_mask = per_query_mask.any(dim=0)
        shared_union_mask = (
            frame_union_mask.clone() if shared_union_mask is None else shared_union_mask | frame_union_mask
        )
        frame_inputs.append(
            {
                "latent": latent,
                "frame_local": frame_local,
                "frame_weights": frame_weights,
                "active_before": len(active_frame_original),
            }
        )

    if shared_union_mask is None or shared_active_spatial is None:
        raise RuntimeError("Future-frame shared spatial selection produced no mask")
    selected_spatial = [
        shared_active_spatial[index]
        for index, selected in enumerate(shared_union_mask.detach().cpu().tolist())
        if bool(selected)
    ]
    frame_rows: list[dict[str, Any]] = []
    for frame_input in frame_inputs:
        frame_local = frame_input["frame_local"]
        frame_weights = frame_input["frame_weights"]
        selected_frame_local = frame_local[shared_union_mask]
        keep_local[selected_frame_local] = True
        selected_mass = frame_weights[:, shared_union_mask].sum(dim=-1)
        total_mass = frame_weights.sum(dim=-1)
        coverage = selected_mass / total_mass.clamp_min(eps)
        frame_rows.append(
            {
                "latent": int(frame_input["latent"]),
                "active_before": int(frame_input["active_before"]),
                "selected_after": len(selected_spatial),
                "selected_spatial_positions": selected_spatial,
                "coverage_min": float(coverage.min().item()),
                "coverage_mean": float(coverage.mean().item()),
            }
        )

    selected_local = torch.nonzero(keep_local, as_tuple=False).flatten()
    selected_original = active_original_positions.index_select(0, selected_local)
    return selected_local, selected_original, frame_rows


def restore_terminal_hidden(*, torch: Any, side_buffer: Any, active_hidden: Any, active_positions: Any) -> Any:
    """Restore the original GEN order without processing dropped tokens again."""

    restored = side_buffer.clone()
    restored.index_copy_(0, active_positions, active_hidden)
    if not bool(torch.isfinite(restored).all()):
        raise RuntimeError("Restored terminal GEN hidden contains NaN/Inf")
    return restored


class ActionAttentionMass90Controller:
    """Request-local controller for the two-pass causal sparsity experiment."""

    def __init__(
        self,
        *,
        torch: Any,
        net: Any,
        guidance: float,
        num_steps: int,
        threshold: float = 0.9,
        output_dir: Path | None = None,
    ) -> None:
        self.torch = torch
        self.net = net
        self.guidance = float(guidance)
        self.num_steps = int(num_steps)
        self.threshold = float(threshold)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold must be in (0,1]")
        self.model = net.language_model.model
        self.layers = list(self.model.layers)
        self._handles: list[Any] = []
        self._current: dict[str, Any] | None = None
        self._layout: dict[str, Any] | None = None
        self._forward_call_index = 0
        self._original_pack: SequencePack | None = None
        self._position_embeddings: tuple[SequencePack, SequencePack] | None = None
        self._active_original_positions: Any | None = None
        self._side_buffer: Any | None = None
        self._probe_decision: tuple[Any, Any, list[dict[str, Any]]] | None = None
        self._block_rows: list[dict[str, Any]] = []
        self._token_rows: list[dict[str, Any]] = []
        self._completed_stacks = 0
        self.stack_active = False

    def __enter__(self) -> "ActionAttentionMass90Controller":
        if hasattr(self.model, "_action_attention_mass90_controller"):
            raise RuntimeError("Action-attention mass90 controller is already installed")
        for layer in self.layers:
            callback = layer.self_attn._attention_stats_capture_callback
            if callback is not None:
                raise RuntimeError("Action-attention mass90 intervention conflicts with an attention capture callback")
        setattr(self.model, "_action_attention_mass90_controller", self)
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
        if getattr(self.model, "_action_attention_mass90_controller", None) is self:
            delattr(self.model, "_action_attention_mass90_controller")
        self._current = None

    def _call_semantics(self, call_index: int) -> tuple[int, str]:
        if self.guidance == 1.0:
            return call_index, "conditional"
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
            raise RuntimeError("GEN/action token layout changed during the intervention")
        timesteps = getattr(getattr(packed_sequence, "vision", None), "timesteps", None)
        timestep = float("nan")
        if self.torch.is_tensor(timesteps) and timesteps.numel():
            timestep = float(timesteps.reshape(-1)[0].detach().float().cpu())
        self._current = {"step": step, "branch": branch, "timestep": timestep}

    def _network_post_hook(self, module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any], output: Any) -> None:
        del module, args, kwargs, output
        if self.stack_active:
            self.abort_stack()
            raise RuntimeError("Network forward ended before sparse stack restoration")
        self._current = None

    def begin_stack(
        self,
        *,
        hidden_states: SequencePack,
        position_embeddings: tuple[SequencePack, SequencePack],
        memory_gen_only: bool,
        natten_metadata_list: list | None,
    ) -> None:
        del memory_gen_only
        if self._current is None:
            return
        if self.stack_active:
            raise RuntimeError("Nested sparse Transformer stacks are unsupported")
        if natten_metadata_list is not None:
            raise RuntimeError("Action-attention mass90 requires dense/two-way attention, not NATTEN")
        if self._layout is None:
            raise RuntimeError("Missing original token layout")
        num_gen = int(hidden_states["_num_full_tokens"])
        if num_gen != int(self._layout["num_gen_tokens"]):
            raise RuntimeError(f"Initial GEN length {num_gen} differs from layout {self._layout['num_gen_tokens']}")
        if bool(hidden_states.get("is_sharded", False)):
            raise RuntimeError("Context-parallel SequencePacks are unsupported by this experiment")
        self._original_pack = hidden_states
        self._position_embeddings = position_embeddings
        self._active_original_positions = self.torch.arange(
            num_gen, dtype=self.torch.long, device=get_gen_seq(hidden_states).device
        )
        self._side_buffer = get_gen_seq(hidden_states).detach().clone()
        self.stack_active = True

    def _capture_probe(
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
        if self._layout is None or self._active_original_positions is None:
            raise RuntimeError("Probe callback has no live sparse layout")
        if self._probe_decision is not None:
            raise RuntimeError(f"Block {layer_index} emitted multiple probe callbacks")
        self._probe_decision = select_future_mass_union(
            torch=self.torch,
            q_gen=q_gen,
            k_ar=k_ar,
            k_gen=k_gen,
            scaling=scaling,
            token_layout=self._layout,
            active_original_positions=self._active_original_positions,
            threshold=self.threshold,
        )

    def _slice_pack_and_rope(self, hidden_states: SequencePack, selected_local: Any) -> tuple[SequencePack, Any]:
        assert self._position_embeddings is not None
        gen_selected = get_gen_seq(hidden_states).index_select(0, selected_local)
        sparse_pack = make_sequence_pack(und_seq=get_und_seq(hidden_states), gen_seq=gen_selected)
        cos_pack, sin_pack = self._position_embeddings
        sparse_cos = make_sequence_pack(
            und_seq=get_und_seq(cos_pack),
            gen_seq=get_gen_seq(cos_pack).index_select(0, selected_local),
        )
        sparse_sin = make_sequence_pack(
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
        if not self.stack_active or self._current is None or self._layout is None:
            raise RuntimeError("run_layer called without an active intervention")
        if self._position_embeddings is None or self._active_original_positions is None or self._side_buffer is None:
            raise RuntimeError("Sparse stack state is incomplete")
        attention = decoder_layer.self_attn
        if attention._attention_stats_capture_callback is not None:
            raise RuntimeError(f"Block {block} attention callback is already occupied")

        # Read-only full probe of every token still active at this block.
        self._probe_decision = None
        attention._attention_stats_capture_callback = self._capture_probe
        try:
            decoder_layer(
                hidden_states,
                attention_mask,
                self._position_embeddings,
                natten_metadata=None,
                memory_value=memory_value,
                gen_only=gen_only,
            )
        finally:
            attention._attention_stats_capture_callback = None
        if self._probe_decision is None:
            raise RuntimeError(f"Block {block} probe did not produce a token decision")
        selected_local, selected_original, frame_rows = self._probe_decision
        self._probe_decision = None

        gen_before = int(self._active_original_positions.numel())
        sparse_input, sparse_position_embeddings = self._slice_pack_and_rope(hidden_states, selected_local)
        actual_output, lbl_metadata, kv_to_store = decoder_layer(
            sparse_input,
            attention_mask,
            sparse_position_embeddings,
            natten_metadata=None,
            memory_value=memory_value,
            gen_only=gen_only,
        )
        actual_gen = get_gen_seq(actual_output)
        if int(actual_gen.shape[0]) != int(selected_original.numel()):
            raise RuntimeError("Committed block output length differs from selected positions")
        self._side_buffer.index_copy_(0, selected_original, actual_gen)
        self._active_original_positions = selected_original
        self._position_embeddings = sparse_position_embeddings

        original_future = 8 * int(self._layout["latent_shape_thw"][1]) * int(self._layout["latent_shape_thw"][2])
        shared_spatial_masks = {tuple(row["selected_spatial_positions"]) for row in frame_rows}
        if len(shared_spatial_masks) != 1:
            raise RuntimeError("Committed future frames do not share one spatial mask")
        future_after = sum(int(row["selected_after"]) for row in frame_rows)
        future_before = sum(int(row["active_before"]) for row in frame_rows)
        shared_spatial_tokens = int(frame_rows[0]["selected_after"])
        if future_after != 8 * shared_spatial_tokens:
            raise RuntimeError("Future token count is not 8 times the shared spatial mask size")
        gen_after = int(selected_original.numel())
        original_gen = int(self._layout["num_gen_tokens"])
        num_ar = (
            int(getattr(memory_value, "und_k_cached", get_und_seq(hidden_states)).shape[1])
            if (getattr(memory_value, "und_k_cached", None) is not None)
            else int(get_und_seq(hidden_states).shape[0])
        )
        current = self._current
        for row in frame_rows:
            original_frame = len(self._layout["latent_positions"][f"L{row['latent']}"])
            self._token_rows.append(
                {
                    **current,
                    "block": int(block),
                    **row,
                    "dropped_this_block": int(row["active_before"] - row["selected_after"]),
                    "saved_vs_original": int(original_frame - row["selected_after"]),
                    "retained_vs_original": float(row["selected_after"] / original_frame),
                }
            )
        work_denom = gen_before * (num_ar + gen_before)
        self._block_rows.append(
            {
                **current,
                "block": int(block),
                "ar_tokens": num_ar,
                "gen_tokens_before": gen_before,
                "gen_tokens_after": gen_after,
                "future_tokens_before": future_before,
                "future_tokens_after": future_after,
                "shared_future_spatial_tokens": shared_spatial_tokens,
                "dropped_this_block": gen_before - gen_after,
                "saved_gen_vs_original": original_gen - gen_after,
                "saved_future_vs_original": original_future - future_after,
                "gen_retained_ratio": gen_after / original_gen,
                "total_b27_style_key_retained_ratio": (num_ar + gen_after) / (num_ar + original_gen),
                "attention_work_ratio_vs_current_probe": (
                    gen_after * (num_ar + gen_after) / work_denom if work_denom else 1.0
                ),
                "finite": bool(self.torch.isfinite(actual_gen).all()),
            }
        )
        return actual_output, lbl_metadata, kv_to_store

    def end_stack(self, hidden_states: SequencePack) -> SequencePack:
        if not self.stack_active or self._original_pack is None:
            raise RuntimeError("end_stack called without a live sparse stack")
        assert self._active_original_positions is not None and self._side_buffer is not None
        restored_gen = restore_terminal_hidden(
            torch=self.torch,
            side_buffer=self._side_buffer,
            active_hidden=get_gen_seq(hidden_states),
            active_positions=self._active_original_positions,
        )
        restored = from_und_gen_splits(get_und_seq(hidden_states), restored_gen, self._original_pack)
        self._completed_stacks += 1
        self.abort_stack()
        return restored

    def abort_stack(self) -> None:
        self.stack_active = False
        self._original_pack = None
        self._position_embeddings = None
        self._active_original_positions = None
        self._side_buffer = None
        self._probe_decision = None

    def finish(self) -> dict[str, Any]:
        expected_stacks = self.num_steps if self.guidance == 1.0 else self.num_steps * 2
        expected_rows = expected_stacks * len(self.layers)
        if self.stack_active:
            raise RuntimeError("Cannot finish with an active Transformer stack")
        if self._completed_stacks != expected_stacks or len(self._block_rows) != expected_rows:
            raise RuntimeError(
                f"Incomplete intervention: stacks={self._completed_stacks}/{expected_stacks}, "
                f"blocks={len(self._block_rows)}/{expected_rows}"
            )
        if not all(row["finite"] for row in self._block_rows):
            raise RuntimeError("Committed sparse block output contains NaN/Inf")
        saved = [int(row["saved_gen_vs_original"]) for row in self._block_rows]
        retained = [float(row["gen_retained_ratio"]) for row in self._block_rows]
        summary = {
            "schema_version": 1,
            "experiment": "action_attention_per_query_per_frame_mass_union_shared_spatial_mask",
            "threshold": self.threshold,
            "probe_output_committed": False,
            "sparse_output_committed": True,
            "dropped_tokens_reenter_later_blocks": False,
            "future_frames_share_spatial_mask": True,
            "terminal_full_grid_restore": "last-valid-hidden side buffer",
            "physical_block_passes": 2,
            "timing_claim": "none; full probe makes this an oracle validation",
            "completed_stacks": self._completed_stacks,
            "block_calls": len(self._block_rows),
            "mean_saved_gen_tokens": sum(saved) / len(saved),
            "max_saved_gen_tokens": max(saved),
            "mean_gen_retained_ratio": sum(retained) / len(retained),
            "final_gen_tokens_by_stack": [
                row["gen_tokens_after"] for row in self._block_rows if row["block"] == len(self.layers) - 1
            ],
            "all_finite": True,
        }
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with (self.output_dir / "token_savings_by_frame.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=TOKEN_FIELDS)
                writer.writeheader()
                writer.writerows(self._token_rows)
            with (self.output_dir / "token_savings_by_block.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=BLOCK_FIELDS)
                writer.writeheader()
                writer.writerows(self._block_rows)
            (self.output_dir / "token_savings_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        return summary
