# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Per-frame action-aligned sparse routing with L1 residual completion."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from cosmos_framework.data.generator.sequence_packing.runtime import (
    from_und_gen_splits,
    get_gen_seq,
    get_und_seq,
)
from cosmos_framework.scripts.robolab_action_attention_l1_residual_intervention import (
    ActionAttentionL1ResidualController,
)
from cosmos_framework.scripts.robolab_action_attention_mass90_intervention import minimum_mass_mask


def select_l2_l8_aligned_frame_mass(
    *,
    torch: Any,
    q_gen: Any,
    k_ar: Any,
    k_gen: Any,
    scaling: float,
    token_layout: Mapping[str, Any],
    threshold: float,
    eps: float = 1e-12,
) -> tuple[Any, Any, list[dict[str, Any]], Any, list[dict[str, Any]]]:
    """Select an independent mask for each frame from its four aligned actions.

    Predicted action horizons 0--31 are grouped four at a time.  Group 0--3
    aligns with L1, 4--7 with L2, ..., and 28--31 with L8, matching action's
    one-token-per-frame packing and vision's temporal compression factor four.
    L1 remains full, while L2--L8 each use their own minimum aggregate-mass set.
    """

    num_gen = int(token_layout["num_gen_tokens"])
    if int(q_gen.shape[0]) != num_gen or int(k_gen.shape[0]) != num_gen:
        raise RuntimeError("Aligned-frame probe requires the complete GEN sequence")
    if q_gen.ndim != 3 or k_ar.ndim != 3 or k_gen.ndim != 3:
        raise RuntimeError("Expected Q/K in [tokens,heads,head_dim]")
    num_q_heads = int(q_gen.shape[1])
    num_kv_heads = int(k_gen.shape[1])
    if num_q_heads % num_kv_heads:
        raise RuntimeError(f"Invalid GQA geometry q_heads={num_q_heads}, kv_heads={num_kv_heads}")

    predicted = [query for query in token_layout["action_queries"] if query["query_role"] == "predicted"]
    horizon_to_position = {int(query["action_horizon"]): int(query["gen_position"]) for query in predicted}
    if sorted(horizon_to_position) != list(range(32)):
        raise RuntimeError(f"Expected predicted action horizons 0..31, got {sorted(horizon_to_position)}")
    action_index = torch.tensor(
        [horizon_to_position[horizon] for horizon in range(32)],
        dtype=torch.long,
        device=q_gen.device,
    )
    q_heads = q_gen.index_select(0, action_index).detach().float().permute(1, 0, 2).contiguous()
    k_all = torch.cat((k_ar, k_gen), dim=0).detach().float()
    k_heads = (
        k_all.repeat_interleave(num_q_heads // num_kv_heads, dim=1).permute(1, 0, 2).contiguous()
    )
    probabilities = torch.softmax(torch.matmul(q_heads, k_heads.transpose(1, 2)) * float(scaling), dim=-1)
    head_mean = probabilities.mean(dim=0)  # [32 action horizons, all keys]
    if not bool(torch.isfinite(head_mean).all()):
        raise RuntimeError("Action attention contains NaN/Inf")

    num_ar = int(k_ar.shape[0])
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
    masks = []
    frame_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    for latent in range(2, 9):
        horizon_start = 4 * (latent - 1)
        horizons = list(range(horizon_start, horizon_start + 4))
        frame_positions = torch.tensor(
            token_layout["latent_positions"][f"L{latent}"],
            dtype=torch.long,
            device=q_gen.device,
        )
        frame_weights = head_mean[horizons].index_select(-1, frame_positions + num_ar)
        aggregate_weights = frame_weights.mean(dim=0)
        frame_mask = minimum_mass_mask(aggregate_weights.unsqueeze(0), threshold).squeeze(0)
        selected_spatial = torch.nonzero(frame_mask, as_tuple=False).flatten()
        if not int(selected_spatial.numel()):
            raise RuntimeError(f"Aligned-frame selection produced an empty L{latent} mask")
        keep[frame_positions.index_select(0, selected_spatial)] = True
        masks.append(frame_mask)

        per_query_coverage = frame_weights[:, frame_mask].sum(dim=-1) / frame_weights.sum(
            dim=-1
        ).clamp_min(eps)
        aggregate_coverage = aggregate_weights[frame_mask].sum() / aggregate_weights.sum().clamp_min(eps)
        raw_frame_mass = frame_weights.sum(dim=-1)
        selected_spatial_list = selected_spatial.detach().cpu().tolist()
        frame_rows.append(
            {
                "latent": latent,
                "selected_after": int(selected_spatial.numel()),
                "selected_spatial_positions": selected_spatial_list,
                "coverage_min": float(per_query_coverage.min().item()),
                "coverage_mean": float(per_query_coverage.mean().item()),
            }
        )
        alignment_rows.append(
            {
                "latent": latent,
                "action_horizon_start": horizons[0],
                "action_horizon_end": horizons[-1],
                "selected_spatial_tokens": int(selected_spatial.numel()),
                "aggregate_coverage": float(aggregate_coverage.item()),
                "per_query_coverage_min": float(per_query_coverage.min().item()),
                "per_query_coverage_mean": float(per_query_coverage.mean().item()),
                "raw_frame_mass_min": float(raw_frame_mass.min().item()),
                "raw_frame_mass_mean": float(raw_frame_mass.mean().item()),
                "raw_frame_mass_max": float(raw_frame_mass.max().item()),
                "selected_spatial_positions": selected_spatial_list,
            }
        )
    selected = torch.nonzero(keep, as_tuple=False).flatten()
    return selected, selected, frame_rows, torch.stack(masks), alignment_rows


def restore_with_per_frame_l1_residual(
    *,
    torch: Any,
    original_gen: Any,
    sparse_output_gen: Any,
    selected_positions: Any,
    token_layout: Mapping[str, Any],
    frame_masks: Any,
) -> tuple[Any, Any, Any, Any]:
    """Restore each L2--L8 omitted set with matching-position L1 residual."""

    if tuple(frame_masks.shape) != (7, len(token_layout["latent_positions"]["L1"])):
        raise RuntimeError(f"Expected frame masks [7,spatial], got {tuple(frame_masks.shape)}")
    if int(sparse_output_gen.shape[0]) != int(selected_positions.numel()):
        raise RuntimeError("Sparse output length differs from selected GEN positions")
    l1_positions = torch.tensor(
        token_layout["latent_positions"]["L1"], dtype=torch.long, device=original_gen.device
    )
    l1_offsets = torch.searchsorted(selected_positions, l1_positions)
    if not bool(torch.equal(selected_positions.index_select(0, l1_offsets), l1_positions)):
        raise RuntimeError("L1 must remain fully computed")
    l1_residual = sparse_output_gen.index_select(0, l1_offsets) - original_gen.index_select(0, l1_positions)

    target_parts = []
    donor_parts = []
    for mask_index, latent in enumerate(range(2, 9)):
        omitted_spatial = torch.nonzero(~frame_masks[mask_index], as_tuple=False).flatten()
        frame_positions = torch.tensor(
            token_layout["latent_positions"][f"L{latent}"],
            dtype=torch.long,
            device=original_gen.device,
        )
        target_parts.append(frame_positions.index_select(0, omitted_spatial))
        donor_parts.append(omitted_spatial)
    targets = torch.cat(target_parts)
    donor_spatial = torch.cat(donor_parts)

    full = original_gen.clone()
    full.index_copy_(0, selected_positions, sparse_output_gen)
    if int(targets.numel()):
        recovered = original_gen.index_select(0, targets) + l1_residual.index_select(0, donor_spatial)
        full.index_copy_(0, targets, recovered)
    return full, targets, donor_spatial, l1_residual


class ActionAttentionAlignedFrameL1ResidualController(ActionAttentionL1ResidualController):
    """Current-block oracle with per-frame aligned action routing."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._probe_frame_masks: Any | None = None
        self._probe_alignment_rows: list[dict[str, Any]] = []
        self._alignment_rows: list[dict[str, Any]] = []

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
        selected_local, selected_original, rows, masks, alignment_rows = (
            select_l2_l8_aligned_frame_mass(
                torch=self.torch,
                q_gen=q_gen,
                k_ar=k_ar,
                k_gen=k_gen,
                scaling=scaling,
                token_layout=self._layout,
                threshold=self.threshold,
            )
        )
        self._probe_decision = (selected_local, selected_original, rows)
        self._probe_frame_masks = masks
        self._probe_alignment_rows = alignment_rows
        self._probe_ar_tokens = int(k_ar.shape[0])

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
            raise RuntimeError("run_layer called without an active aligned-frame intervention")
        if self._position_embeddings is None:
            raise RuntimeError("Aligned-frame stack state is incomplete")
        original_gen = get_gen_seq(hidden_states)
        original_tokens = int(self._layout["num_gen_tokens"])
        if int(original_gen.shape[0]) != original_tokens:
            raise RuntimeError(f"Block {block} did not receive the complete GEN sequence")
        attention = decoder_layer.self_attn
        if attention._attention_stats_capture_callback is not None:
            raise RuntimeError(f"Block {block} attention callback is already occupied")

        self._probe_decision = None
        self._probe_frame_masks = None
        self._probe_alignment_rows = []
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
        if self._probe_decision is None or self._probe_frame_masks is None:
            raise RuntimeError(f"Block {block} full probe did not produce aligned-frame masks")
        selected_local, selected_original, frame_rows = self._probe_decision
        frame_masks = self._probe_frame_masks

        sparse_input, sparse_rope = self._slice_pack_and_rope(hidden_states, selected_local)
        sparse_output, lbl_metadata, kv_to_store = decoder_layer(
            sparse_input,
            attention_mask,
            sparse_rope,
            natten_metadata=None,
            memory_value=memory_value,
            gen_only=gen_only,
        )
        full_gen, targets, donor_spatial, l1_residual = restore_with_per_frame_l1_residual(
            torch=self.torch,
            original_gen=original_gen,
            sparse_output_gen=get_gen_seq(sparse_output),
            selected_positions=selected_original,
            token_layout=self._layout,
            frame_masks=frame_masks,
        )
        full_output = from_und_gen_splits(get_und_seq(sparse_output), full_gen, hidden_states)
        expected = original_gen.index_select(0, targets) + l1_residual.index_select(0, donor_spatial)
        exact = bool(self.torch.equal(full_gen.index_select(0, targets), expected))
        finite = bool(self.torch.isfinite(full_gen).all())
        if not exact or not finite:
            raise RuntimeError(f"Block {block} aligned-frame L1-residual completion failed")

        current = self._current
        spatial_tokens = len(self._layout["latent_positions"]["L1"])
        selected_counts = [int(mask.sum().item()) for mask in frame_masks]
        for row in frame_rows:
            selected_count = int(row["selected_after"])
            self._token_rows.append(
                {
                    **current,
                    "block": block,
                    **row,
                    "active_before": spatial_tokens,
                    "dropped_this_block": spatial_tokens - selected_count,
                    "saved_vs_original": spatial_tokens - selected_count,
                    "retained_vs_original": selected_count / spatial_tokens,
                }
            )
        for row in self._probe_alignment_rows:
            self._alignment_rows.append({**current, "block": block, **row})

        sparse_tokens = int(selected_original.numel())
        self._block_rows.append(
            {
                **current,
                "block": block,
                "ar_tokens": self._probe_ar_tokens,
                "probe_gen_tokens": original_tokens,
                "sparse_gen_tokens": sparse_tokens,
                "restored_gen_tokens": int(full_gen.shape[0]),
                "l1_full_tokens": spatial_tokens,
                "shared_l2_l8_spatial_tokens": sum(selected_counts) / len(selected_counts),
                "computed_l2_l8_tokens": sum(selected_counts),
                "l1_residual_recovered_tokens": int(targets.numel()),
                "saved_sparse_gen_tokens": original_tokens - sparse_tokens,
                "sparse_gen_retained_ratio": sparse_tokens / original_tokens,
                "residual_completion_exact": exact,
                "finite": finite,
            }
        )
        self._probe_decision = None
        self._probe_frame_masks = None
        self._probe_alignment_rows = []
        self._probe_ar_tokens = None
        return full_output, lbl_metadata, kv_to_store

    def finish(self) -> dict[str, Any]:
        summary = super().finish()
        selected = [int(row["selected_spatial_tokens"]) for row in self._alignment_rows]
        summary.update(
            {
                "experiment": "current_block_action_aligned_frame_mass90_l1_residual_completion",
                "mask_selection": "independent L2-L8 masks from four temporally aligned action horizons",
                "action_horizon_groups": {f"L{latent}": [4 * (latent - 1), 4 * latent - 1] for latent in range(2, 9)},
                "l2_l8_share_spatial_mask": False,
                "per_frame_masks_shared": False,
                "mean_selected_l2_l8_spatial_tokens": sum(selected) / len(selected),
                "min_selected_l2_l8_spatial_tokens": min(selected),
                "max_selected_l2_l8_spatial_tokens": max(selected),
            }
        )
        if self.output_dir is not None:
            output_dir = Path(self.output_dir)
            fields = [
                "step",
                "timestep",
                "branch",
                "block",
                "latent",
                "action_horizon_start",
                "action_horizon_end",
                "selected_spatial_tokens",
                "aggregate_coverage",
                "per_query_coverage_min",
                "per_query_coverage_mean",
                "raw_frame_mass_min",
                "raw_frame_mass_mean",
                "raw_frame_mass_max",
                "selected_spatial_positions",
            ]
            with (output_dir / "aligned_frame_selection.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self._alignment_rows)
            (output_dir / "token_savings_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        return summary
