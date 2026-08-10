# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Same-block action-attention oracle with L1 residual completion.

Every decoder block first runs a full read-only probe.  The probe's exact
post-RoPE Action-Q/Future-K attention chooses a variable shared spatial mask
for L2..L8.  The same block input is then re-run with L0, L1 and all action
tokens intact plus the selected L2..L8 positions.  Missing L2..L8 positions
receive the committed L1 residual at the matching spatial coordinate, so the
next block always receives the complete GEN sequence.
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
)
from cosmos_framework.scripts.robolab_action_attention_mass90_intervention import (
    ActionAttentionMass90Controller,
    minimum_mass_mask,
)


def select_l2_l8_shared_mass_union(
    *,
    torch: Any,
    q_gen: Any,
    k_ar: Any,
    k_gen: Any,
    scaling: float,
    token_layout: Mapping[str, Any],
    threshold: float,
    eps: float = 1e-12,
) -> tuple[Any, Any, list[dict[str, Any]], Any]:
    """Select one variable spatial mask for L2..L8 from the full probe.

    For every predicted action query and each frame L2..L8, the smallest set
    reaching ``threshold`` of that frame's attention mass is computed.  Their
    union in spatial-coordinate space is applied identically to all L2..L8.
    L0, L1, q0 and q1..q32 are mandatory.
    """

    num_gen = int(token_layout["num_gen_tokens"])
    if int(q_gen.shape[0]) != num_gen or int(k_gen.shape[0]) != num_gen:
        raise RuntimeError("L1-residual probe requires the complete GEN sequence")
    if q_gen.ndim != 3 or k_ar.ndim != 3 or k_gen.ndim != 3:
        raise RuntimeError("Expected Q/K in [tokens,heads,head_dim]")
    num_q_heads = int(q_gen.shape[1])
    num_kv_heads = int(k_gen.shape[1])
    if num_q_heads % num_kv_heads:
        raise RuntimeError(f"Invalid GQA geometry q_heads={num_q_heads}, kv_heads={num_kv_heads}")

    predicted = [
        int(query["gen_position"])
        for query in token_layout["action_queries"]
        if query["query_role"] == "predicted"
    ]
    if len(predicted) != 32:
        raise RuntimeError(f"Expected 32 predicted Action Queries, got {len(predicted)}")
    action_index = torch.tensor(predicted, dtype=torch.long, device=q_gen.device)
    q_heads = q_gen.index_select(0, action_index).detach().float().permute(1, 0, 2).contiguous()
    k_all = torch.cat((k_ar, k_gen), dim=0).detach().float()
    k_heads = (
        k_all.repeat_interleave(num_q_heads // num_kv_heads, dim=1).permute(1, 0, 2).contiguous()
    )
    probabilities = torch.softmax(torch.matmul(q_heads, k_heads.transpose(1, 2)) * float(scaling), dim=-1)
    head_mean = probabilities.mean(dim=0)
    if not bool(torch.isfinite(head_mean).all()):
        raise RuntimeError("Action attention contains NaN/Inf")

    num_ar = int(k_ar.shape[0])
    shared_mask = None
    frame_inputs: list[tuple[int, Any, Any]] = []
    for latent in range(2, 9):
        frame_positions = torch.tensor(
            token_layout["latent_positions"][f"L{latent}"],
            dtype=torch.long,
            device=q_gen.device,
        )
        frame_weights = head_mean.index_select(-1, frame_positions + num_ar)
        per_query_mask = minimum_mass_mask(frame_weights, threshold)
        frame_mask = per_query_mask.any(dim=0)
        shared_mask = frame_mask.clone() if shared_mask is None else shared_mask | frame_mask
        frame_inputs.append((latent, frame_positions, frame_weights))
    if shared_mask is None:
        raise RuntimeError("L2..L8 selection produced no shared mask")

    l2_l8_positions = torch.tensor(
        [
            int(position)
            for latent in range(2, 9)
            for position in token_layout["latent_positions"][f"L{latent}"]
        ],
        dtype=torch.long,
        device=q_gen.device,
    )
    keep = torch.ones(num_gen, dtype=torch.bool, device=q_gen.device)
    keep[l2_l8_positions] = False
    rows: list[dict[str, Any]] = []
    selected_spatial = torch.nonzero(shared_mask, as_tuple=False).flatten()
    selected_spatial_list = selected_spatial.detach().cpu().tolist()
    for latent, frame_positions, frame_weights in frame_inputs:
        selected_positions = frame_positions.index_select(0, selected_spatial)
        keep[selected_positions] = True
        selected_mass = frame_weights.index_select(-1, selected_spatial).sum(dim=-1)
        total_mass = frame_weights.sum(dim=-1)
        coverage = selected_mass / total_mass.clamp_min(eps)
        rows.append(
            {
                "latent": latent,
                "selected_after": int(selected_spatial.numel()),
                "selected_spatial_positions": selected_spatial_list,
                "coverage_min": float(coverage.min().item()),
                "coverage_mean": float(coverage.mean().item()),
            }
        )
    selected = torch.nonzero(keep, as_tuple=False).flatten()
    return selected, selected, rows, shared_mask


def restore_with_l1_residual(
    *,
    torch: Any,
    original_gen: Any,
    sparse_output_gen: Any,
    selected_positions: Any,
    token_layout: Mapping[str, Any],
    shared_mask: Any,
) -> tuple[Any, Any, Any]:
    """Scatter sparse output and fill omitted L2..L8 with direct L1 residual."""

    if int(sparse_output_gen.shape[0]) != int(selected_positions.numel()):
        raise RuntimeError("Sparse output length differs from selected GEN positions")
    l1_positions = torch.tensor(
        token_layout["latent_positions"]["L1"], dtype=torch.long, device=original_gen.device
    )
    l1_offsets = torch.searchsorted(selected_positions, l1_positions)
    if not bool(torch.equal(selected_positions.index_select(0, l1_offsets), l1_positions)):
        raise RuntimeError("L1 must remain fully computed in the committed sparse pass")
    l1_residual = sparse_output_gen.index_select(0, l1_offsets) - original_gen.index_select(0, l1_positions)

    omitted_spatial = torch.nonzero(~shared_mask, as_tuple=False).flatten()
    target_parts = []
    for latent in range(2, 9):
        frame = torch.tensor(
            token_layout["latent_positions"][f"L{latent}"],
            dtype=torch.long,
            device=original_gen.device,
        )
        target_parts.append(frame.index_select(0, omitted_spatial))
    targets = torch.cat(target_parts) if target_parts else selected_positions.new_empty(0)
    donor_spatial = omitted_spatial.repeat(7)

    full = original_gen.clone()
    full.index_copy_(0, selected_positions, sparse_output_gen)
    if int(targets.numel()):
        recovered = original_gen.index_select(0, targets) + l1_residual.index_select(0, donor_spatial)
        full.index_copy_(0, targets, recovered)
    return full, targets, l1_residual


class ActionAttentionL1ResidualController(ActionAttentionMass90Controller):
    """Current-block oracle whose committed output is complete after L1 reuse."""

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
        super().__init__(
            torch=torch,
            net=net,
            guidance=guidance,
            num_steps=num_steps,
            threshold=threshold,
            output_dir=output_dir,
        )
        self._probe_shared_mask: Any | None = None
        self._probe_ar_tokens: int | None = None

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
            raise RuntimeError("L1-residual oracle requires dense/two-way attention")
        if self._layout is None:
            raise RuntimeError("Missing original token layout")
        num_gen = int(hidden_states["_num_full_tokens"])
        if num_gen != int(self._layout["num_gen_tokens"]):
            raise RuntimeError("Initial GEN sequence is not complete")
        if bool(hidden_states.get("is_sharded", False)):
            raise RuntimeError("Context-parallel SequencePacks are unsupported")
        self._original_pack = hidden_states
        self._position_embeddings = position_embeddings
        self._active_original_positions = self.torch.arange(
            num_gen, dtype=self.torch.long, device=get_gen_seq(hidden_states).device
        )
        self._side_buffer = None
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
        if self._layout is None:
            raise RuntimeError("Probe callback has no token layout")
        if self._probe_decision is not None:
            raise RuntimeError(f"Block {layer_index} emitted multiple probe callbacks")
        selected_local, selected_original, rows, shared_mask = select_l2_l8_shared_mass_union(
            torch=self.torch,
            q_gen=q_gen,
            k_ar=k_ar,
            k_gen=k_gen,
            scaling=scaling,
            token_layout=self._layout,
            threshold=self.threshold,
        )
        self._probe_decision = (selected_local, selected_original, rows)
        self._probe_shared_mask = shared_mask
        self._probe_ar_tokens = int(k_ar.shape[0])

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
            raise RuntimeError("run_layer called without an active L1-residual intervention")
        if self._position_embeddings is None or self._active_original_positions is None:
            raise RuntimeError("L1-residual stack state is incomplete")
        original_gen = get_gen_seq(hidden_states)
        original_tokens = int(self._layout["num_gen_tokens"])
        if int(original_gen.shape[0]) != original_tokens:
            raise RuntimeError(f"Block {block} did not receive the complete GEN sequence")
        attention = decoder_layer.self_attn
        if attention._attention_stats_capture_callback is not None:
            raise RuntimeError(f"Block {block} attention callback is already occupied")

        self._probe_decision = None
        self._probe_shared_mask = None
        self._probe_ar_tokens = None
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
        if self._probe_decision is None or self._probe_shared_mask is None or self._probe_ar_tokens is None:
            raise RuntimeError(f"Block {block} full probe did not produce a sparse decision")
        selected_local, selected_original, frame_rows = self._probe_decision
        shared_mask = self._probe_shared_mask

        sparse_input, sparse_rope = self._slice_pack_and_rope(hidden_states, selected_local)
        sparse_output, lbl_metadata, kv_to_store = decoder_layer(
            sparse_input,
            attention_mask,
            sparse_rope,
            natten_metadata=None,
            memory_value=memory_value,
            gen_only=gen_only,
        )
        sparse_gen = get_gen_seq(sparse_output)
        full_gen, targets, l1_residual = restore_with_l1_residual(
            torch=self.torch,
            original_gen=original_gen,
            sparse_output_gen=sparse_gen,
            selected_positions=selected_original,
            token_layout=self._layout,
            shared_mask=shared_mask,
        )
        full_output = from_und_gen_splits(get_und_seq(sparse_output), full_gen, hidden_states)
        omitted_spatial = self.torch.nonzero(~shared_mask, as_tuple=False).flatten()
        expected = original_gen.index_select(0, targets) + l1_residual.index_select(
            0, omitted_spatial.repeat(7)
        )
        exact = bool(self.torch.equal(full_gen.index_select(0, targets), expected))
        finite = bool(self.torch.isfinite(full_gen).all())
        if not exact or not finite:
            raise RuntimeError(f"Block {block} L1-residual completion failed")

        spatial_tokens = len(self._layout["latent_positions"]["L1"])
        shared_tokens = int(shared_mask.sum().item())
        sparse_tokens = int(selected_original.numel())
        current = self._current
        for row in frame_rows:
            self._token_rows.append(
                {
                    **current,
                    "block": block,
                    **row,
                    "active_before": spatial_tokens,
                    "dropped_this_block": spatial_tokens - shared_tokens,
                    "saved_vs_original": spatial_tokens - shared_tokens,
                    "retained_vs_original": shared_tokens / spatial_tokens,
                }
            )
        self._block_rows.append(
            {
                **current,
                "block": block,
                "ar_tokens": self._probe_ar_tokens,
                "probe_gen_tokens": original_tokens,
                "sparse_gen_tokens": sparse_tokens,
                "restored_gen_tokens": int(full_gen.shape[0]),
                "l1_full_tokens": spatial_tokens,
                "shared_l2_l8_spatial_tokens": shared_tokens,
                "computed_l2_l8_tokens": 7 * shared_tokens,
                "l1_residual_recovered_tokens": int(targets.numel()),
                "saved_sparse_gen_tokens": original_tokens - sparse_tokens,
                "sparse_gen_retained_ratio": sparse_tokens / original_tokens,
                "residual_completion_exact": exact,
                "finite": finite,
            }
        )
        self._probe_decision = None
        self._probe_shared_mask = None
        self._probe_ar_tokens = None
        return full_output, lbl_metadata, kv_to_store

    def end_stack(self, hidden_states: SequencePack) -> SequencePack:
        if not self.stack_active or self._layout is None:
            raise RuntimeError("end_stack called without an active L1-residual stack")
        if int(get_gen_seq(hidden_states).shape[0]) != int(self._layout["num_gen_tokens"]):
            raise RuntimeError("L1-residual stack did not finish with a complete GEN sequence")
        self._completed_stacks += 1
        self.abort_stack()
        return hidden_states

    def finish(self) -> dict[str, Any]:
        expected_stacks = self.num_steps if self.guidance == 1.0 else self.num_steps * 2
        expected_rows = expected_stacks * len(self.layers)
        if self.stack_active:
            raise RuntimeError("Cannot finish with an active Transformer stack")
        if self._completed_stacks != expected_stacks or len(self._block_rows) != expected_rows:
            raise RuntimeError(
                f"Incomplete intervention stacks={self._completed_stacks}/{expected_stacks}, "
                f"blocks={len(self._block_rows)}/{expected_rows}"
            )
        if not all(row["finite"] and row["residual_completion_exact"] for row in self._block_rows):
            raise RuntimeError("L1-residual output failed finite/exact validation")
        saved = [int(row["saved_sparse_gen_tokens"]) for row in self._block_rows]
        retained = [float(row["sparse_gen_retained_ratio"]) for row in self._block_rows]
        shared = [int(row["shared_l2_l8_spatial_tokens"]) for row in self._block_rows]
        summary = {
            "schema_version": 1,
            "experiment": "current_block_action_mass90_l1_residual_completion",
            "threshold": self.threshold,
            "physical_block_passes": 2,
            "probe_input_full": True,
            "probe_output_committed": False,
            "sparse_output_committed": True,
            "l1_always_full": True,
            "l2_l8_share_spatial_mask": True,
            "shared_spatial_tokens_fixed": False,
            "missing_l2_l8_completion": "direct same-block L1 residual at matching spatial position",
            "next_block_receives_full_gen": True,
            "dropped_tokens_reenter_next_block": True,
            "timing_claim": "none; every block includes full probe plus committed sparse pass",
            "completed_stacks": self._completed_stacks,
            "block_calls": len(self._block_rows),
            "mean_saved_sparse_gen_tokens": sum(saved) / len(saved),
            # Compatibility aliases consumed by the existing oracle server.
            "mean_saved_gen_tokens": sum(saved) / len(saved),
            "max_saved_sparse_gen_tokens": max(saved),
            "mean_sparse_gen_retained_ratio": sum(retained) / len(retained),
            "mean_shared_l2_l8_spatial_tokens": sum(shared) / len(shared),
            "min_shared_l2_l8_spatial_tokens": min(shared),
            "max_shared_l2_l8_spatial_tokens": max(shared),
            "final_sparse_gen_tokens_by_stack": [
                row["sparse_gen_tokens"] for row in self._block_rows if row["block"] == len(self.layers) - 1
            ],
            "final_gen_tokens_by_stack": [
                row["sparse_gen_tokens"] for row in self._block_rows if row["block"] == len(self.layers) - 1
            ],
            "all_finite": True,
        }
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            token_fields = [
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
            block_fields = [
                "step",
                "timestep",
                "branch",
                "block",
                "ar_tokens",
                "probe_gen_tokens",
                "sparse_gen_tokens",
                "restored_gen_tokens",
                "l1_full_tokens",
                "shared_l2_l8_spatial_tokens",
                "computed_l2_l8_tokens",
                "l1_residual_recovered_tokens",
                "saved_sparse_gen_tokens",
                "sparse_gen_retained_ratio",
                "residual_completion_exact",
                "finite",
            ]
            with (self.output_dir / "token_savings_by_frame.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=token_fields)
                writer.writeheader()
                writer.writerows(self._token_rows)
            with (self.output_dir / "token_savings_by_block.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=block_fields)
                writer.writeheader()
                writer.writerows(self._block_rows)
            (self.output_dir / "token_savings_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        return summary
