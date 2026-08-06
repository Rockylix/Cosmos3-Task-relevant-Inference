# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Online grouped GEN-attention and value statistics for RoboLab experiments.

The collector is default-off and read-only.  It recomputes the dense attention
probabilities in small FP32 query chunks from the exact post-norm/post-RoPE Q/K
tensors and observes the exact post-projection/head-reshape V tensors used by
the model. It writes only aggregated statistics and never changes the attention
kernel or its output.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

SUMMARY_FIELDS = ("mean", "std", "p10", "p50", "p90", "min", "max")
GROUP_FIELDS = [
    "shift",
    "task",
    "chunk",
    "branch",
    "step",
    "timestep",
    "block",
    "query_latent",
    "head",
    "key_group",
    "num_query_tokens",
    "num_key_tokens",
    "total_key_tokens",
    "baseline_fraction",
    "value_capacity_fraction",
] + [
    f"{metric}_{stat}"
    for metric in (
        "mass",
        "enrichment",
        "log2_enrichment",
        "value_weighted_magnitude",
        "value_weighted_share",
        "value_weighted_enrichment",
        "value_reweight_ratio",
        "group_output_norm",
        "cancellation_ratio",
        "direction_contribution",
    )
    for stat in SUMMARY_FIELDS
]
VALUE_FIELDS = [
    "shift",
    "task",
    "chunk",
    "branch",
    "step",
    "timestep",
    "block",
    "head",
    "kv_head",
    "gqa_repeat_factor",
    "key_group",
    "num_key_tokens",
    "total_key_tokens",
    "value_capacity_fraction",
] + [f"{metric}_{stat}" for metric in ("value_l2", "value_rms") for stat in SUMMARY_FIELDS]
ENTROPY_FIELDS = [
    "shift",
    "task",
    "chunk",
    "branch",
    "step",
    "timestep",
    "block",
    "query_latent",
    "head",
    "num_query_tokens",
    "total_key_tokens",
] + [f"entropy_{stat}" for stat in SUMMARY_FIELDS]
CALL_FIELDS = [
    "shift",
    "task",
    "chunk",
    "branch",
    "step",
    "timestep",
    "block",
    "num_ar_tokens",
    "num_gen_tokens",
    "total_key_tokens",
    "num_q_heads",
    "num_kv_heads",
    "gqa_repeat_factor",
    "head_dim",
]


def _summary(torch: Any, values: Any) -> dict[str, float]:
    values = values.detach().float().cpu()
    if values.numel() == 0:
        raise RuntimeError("Cannot summarize an empty tensor")
    quantiles = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9]))
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "p10": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _prefix_summary(prefix: str, values: Any, torch: Any) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in _summary(torch, values).items()}


def _gen_token_layout(torch: Any, packed_sequence: Any) -> dict[str, Any]:
    full_indexes: list[int] = []
    offset = 0
    for mode, split_len in zip(packed_sequence.attn_modes, packed_sequence.split_lens, strict=True):
        split_len = int(split_len)
        if mode == "full":
            full_indexes.extend(range(offset, offset + split_len))
        offset += split_len
    full_position = {original: position for position, original in enumerate(full_indexes)}

    vision = getattr(packed_sequence, "vision", None)
    action = getattr(packed_sequence, "action", None)
    vision_indexes = getattr(vision, "sequence_indexes", None)
    if vision is None or not torch.is_tensor(vision_indexes):
        raise RuntimeError("Attention statistics require packed vision tokens")
    vision_original = [int(value) for value in vision_indexes.detach().cpu().tolist()]
    action_indexes = getattr(action, "sequence_indexes", None)
    action_original = (
        [int(value) for value in action_indexes.detach().cpu().tolist()] if torch.is_tensor(action_indexes) else []
    )
    try:
        vision_positions = [full_position[index] for index in vision_original]
        action_positions = [full_position[index] for index in action_original]
    except KeyError as exc:
        raise RuntimeError(f"Modality token {int(exc.args[0])} is not in the GEN sequence") from exc

    token_shapes = getattr(vision, "token_shapes", None)
    if not token_shapes or len(token_shapes) != 1:
        raise RuntimeError(f"Expected exactly one vision token shape, got {token_shapes}")
    latent_shape = tuple(int(value) for value in token_shapes[0])
    if len(latent_shape) != 3 or math.prod(latent_shape) != len(vision_positions):
        raise RuntimeError(f"Vision token shape {latent_shape} does not match {len(vision_positions)} tokens")
    num_latents, height, width = latent_shape
    if num_latents != 9:
        raise RuntimeError(f"Expected condition L0 plus future L1..L8, got {num_latents} vision latents")
    spatial_tokens = height * width
    latent_positions = {
        f"L{latent}": vision_positions[latent * spatial_tokens : (latent + 1) * spatial_tokens]
        for latent in range(num_latents)
    }
    covered = [position for positions in latent_positions.values() for position in positions] + action_positions
    if len(covered) != len(full_indexes) or sorted(covered) != list(range(len(full_indexes))):
        raise RuntimeError("L0..L8 and action groups do not exactly partition GEN tokens")
    return {
        "num_gen_tokens": len(full_indexes),
        "latent_shape_thw": list(latent_shape),
        "latent_positions": latent_positions,
        "action_positions": action_positions,
    }


def compute_group_attention_statistics(
    *,
    torch: Any,
    q_gen: Any,
    k_ar: Any,
    k_gen: Any,
    v_ar: Any,
    v_gen: Any,
    attn_output_gen: Any,
    scaling: float,
    token_layout: Mapping[str, Any],
    query_chunk_size: int = 64,
    eps: float = 1e-12,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    """Compute per-latent/head attention, value-aware, and entropy summaries.

    Every value is first computed for one query token and one Q head.
    Aggregation across the 340 spatial query tokens occurs only after those
    individual values exist.
    """

    tensors = {"q_gen": q_gen, "k_ar": k_ar, "k_gen": k_gen, "v_ar": v_ar, "v_gen": v_gen}
    if any(tensor.ndim != 3 for tensor in tensors.values()):
        raise RuntimeError("q_gen, k_ar/k_gen, and v_ar/v_gen must be [tokens,heads,head_dim]")
    if attn_output_gen.ndim != 3:
        raise RuntimeError("attn_output_gen must be [tokens,q_heads,head_dim]")
    num_gen_tokens = int(token_layout["num_gen_tokens"])
    if q_gen.shape[0] != num_gen_tokens or k_gen.shape[0] != num_gen_tokens:
        raise RuntimeError(
            f"GEN Q/K length mismatch: layout={num_gen_tokens}, q={q_gen.shape[0]}, k={k_gen.shape[0]}"
        )
    if q_gen.shape[2] != k_gen.shape[2] or k_ar.shape[2] != q_gen.shape[2]:
        raise RuntimeError("Q/K head dimensions differ")
    if v_ar.shape != k_ar.shape or v_gen.shape != k_gen.shape:
        raise RuntimeError(
            f"K/V geometry differs: k_ar={tuple(k_ar.shape)}, v_ar={tuple(v_ar.shape)}, "
            f"k_gen={tuple(k_gen.shape)}, v_gen={tuple(v_gen.shape)}"
        )
    if attn_output_gen.shape != q_gen.shape:
        raise RuntimeError(
            f"Actual attention output shape differs from Q: output={tuple(attn_output_gen.shape)}, "
            f"q={tuple(q_gen.shape)}"
        )
    num_q_heads = int(q_gen.shape[1])
    num_kv_heads = int(k_gen.shape[1])
    if int(k_ar.shape[1]) != num_kv_heads or num_q_heads % num_kv_heads != 0:
        raise RuntimeError(f"Invalid GQA geometry: q_heads={num_q_heads}, kv_heads={num_kv_heads}")

    gqa_repeat_factor = num_q_heads // num_kv_heads
    k_all = torch.cat([k_ar, k_gen], dim=0).detach().float()
    v_all = torch.cat([v_ar, v_gen], dim=0).detach().float()
    # Match the model's GQA convention: consecutive groups of Q heads share one
    # KV head, i.e. h_v = floor(h_q / gqa_repeat_factor).
    k_heads = k_all.repeat_interleave(gqa_repeat_factor, dim=1).permute(1, 0, 2).contiguous()
    v_heads = v_all.repeat_interleave(gqa_repeat_factor, dim=1).permute(1, 0, 2).contiguous()
    value_l2 = torch.linalg.vector_norm(v_heads, dim=-1)  # [q_head,key_token]
    value_rms = value_l2 / math.sqrt(int(v_heads.shape[-1]))
    total_keys = int(k_all.shape[0])
    n_ar = int(k_ar.shape[0])
    latent_positions = token_layout["latent_positions"]
    action_positions = token_layout["action_positions"]
    group_positions: list[tuple[str, list[int]]] = [("K_AR", list(range(n_ar)))]
    for latent in range(9):
        group_positions.append((f"K_L{latent}", [n_ar + int(pos) for pos in latent_positions[f"L{latent}"]]))
    group_positions.append(("K_action", [n_ar + int(pos) for pos in action_positions]))

    flat_positions = [position for _, positions in group_positions for position in positions]
    partition_ok = len(flat_positions) == total_keys and sorted(flat_positions) == list(range(total_keys))
    if not partition_ok:
        raise RuntimeError("K_AR, K_L0..K_L8, and K_action do not exactly partition actual attention keys")
    group_indexes = [torch.tensor(positions, dtype=torch.long, device=q_gen.device) for _, positions in group_positions]
    value_capacity = torch.stack(
        [value_l2.index_select(-1, indexes).sum(-1) for indexes in group_indexes], dim=-1
    )
    value_capacity_fraction = value_capacity / value_l2.sum(-1, keepdim=True).clamp_min(eps)
    log_total_keys = math.log(total_keys)
    if log_total_keys <= 0:
        raise RuntimeError(f"Normalized entropy requires at least two keys, got {total_keys}")

    group_rows: list[dict[str, Any]] = []
    entropy_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    mass_sum_error_max = 0.0
    enrichment_identity_error_max = 0.0
    entropy_min = float("inf")
    entropy_max = float("-inf")
    finite_ok = bool(torch.isfinite(value_l2).all()) and bool(torch.isfinite(value_capacity_fraction).all())
    dense_reference_error_max = 0.0
    value_share_sum_error_max = 0.0
    value_enrichment_identity_error_max = 0.0
    value_reweight_identity_error_max = 0.0
    alpha_sum_error_max = 0.0
    cancellation_ratio_min = float("inf")
    cancellation_ratio_max = float("-inf")
    actual_output_relative_l2_max = 0.0
    actual_output_cosine_min = 1.0
    actual_output_max_absolute_error = 0.0

    for head in range(num_q_heads):
        kv_head = head // gqa_repeat_factor
        for group_slot, (group_name, positions) in enumerate(group_positions):
            indexes = group_indexes[group_slot]
            group_value_l2 = value_l2[head].index_select(0, indexes)
            group_value_rms = value_rms[head].index_select(0, indexes)
            value_rows.append(
                {
                    "head": head,
                    "kv_head": kv_head,
                    "gqa_repeat_factor": gqa_repeat_factor,
                    "key_group": group_name,
                    "num_key_tokens": len(positions),
                    "total_key_tokens": total_keys,
                    "value_capacity_fraction": float(value_capacity_fraction[head, group_slot]),
                    **_prefix_summary("value_l2", group_value_l2, torch),
                    **_prefix_summary("value_rms", group_value_rms, torch),
                }
            )

    for latent in range(1, 9):
        query_positions = [int(value) for value in latent_positions[f"L{latent}"]]
        masses_chunks: list[Any] = []
        value_magnitude_chunks: list[Any] = []
        value_share_chunks: list[Any] = []
        group_output_norm_chunks: list[Any] = []
        cancellation_ratio_chunks: list[Any] = []
        direction_contribution_chunks: list[Any] = []
        entropy_chunks: list[Any] = []
        for start in range(0, len(query_positions), query_chunk_size):
            positions = query_positions[start : start + query_chunk_size]
            query_index = torch.tensor(positions, dtype=torch.long, device=q_gen.device)
            q_heads = q_gen.index_select(0, query_index).detach().float().permute(1, 0, 2).contiguous()
            logits = torch.bmm(q_heads, k_heads.transpose(1, 2)) * float(scaling)
            probs = torch.softmax(logits, dim=-1)
            masses = torch.stack([probs.index_select(-1, indexes).sum(-1) for indexes in group_indexes], dim=-1)
            weighted = probs * value_l2[:, None, :]
            value_magnitude = torch.stack(
                [weighted.index_select(-1, indexes).sum(-1) for indexes in group_indexes], dim=-1
            )
            value_share = value_magnitude / weighted.sum(-1, keepdim=True).clamp_min(eps)
            group_output = torch.stack(
                [
                    torch.bmm(
                        probs.index_select(-1, indexes),
                        v_heads.index_select(1, indexes),
                    )
                    for indexes in group_indexes
                ],
                dim=2,
            )  # [q_head,query_token,key_group,head_dim]
            total_output = group_output.sum(dim=2)
            group_output_norm = torch.linalg.vector_norm(group_output, dim=-1)
            cancellation_ratio = group_output_norm / value_magnitude.clamp_min(eps)
            total_output_energy = total_output.square().sum(dim=-1, keepdim=True)
            direction_contribution = (
                (group_output * total_output.unsqueeze(2)).sum(dim=-1)
                / total_output_energy.clamp_min(eps)
            )
            entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(-1) / log_total_keys
            masses_chunks.append(masses.permute(1, 0, 2).detach().cpu())
            value_magnitude_chunks.append(value_magnitude.permute(1, 0, 2).detach().cpu())
            value_share_chunks.append(value_share.permute(1, 0, 2).detach().cpu())
            group_output_norm_chunks.append(group_output_norm.permute(1, 0, 2).detach().cpu())
            cancellation_ratio_chunks.append(cancellation_ratio.permute(1, 0, 2).detach().cpu())
            direction_contribution_chunks.append(direction_contribution.permute(1, 0, 2).detach().cpu())
            entropy_chunks.append(entropy.transpose(0, 1).detach().cpu())

            mass_sum_error_max = max(mass_sum_error_max, float((masses.sum(-1) - 1.0).abs().max()))
            value_share_sum_error_max = max(
                value_share_sum_error_max, float((value_share.sum(-1) - 1.0).abs().max())
            )
            alpha_sum_error_max = max(
                alpha_sum_error_max, float((direction_contribution.sum(-1) - 1.0).abs().max())
            )
            cancellation_ratio_min = min(cancellation_ratio_min, float(cancellation_ratio.min()))
            cancellation_ratio_max = max(cancellation_ratio_max, float(cancellation_ratio.max()))
            actual_output = (
                attn_output_gen.index_select(0, query_index).detach().float().permute(1, 0, 2).contiguous()
            )
            output_diff = total_output - actual_output
            output_relative_l2 = torch.linalg.vector_norm(output_diff) / torch.linalg.vector_norm(
                actual_output
            ).clamp_min(eps)
            output_cosine = torch.nn.functional.cosine_similarity(
                total_output.reshape(-1), actual_output.reshape(-1), dim=0, eps=eps
            )
            actual_output_relative_l2_max = max(actual_output_relative_l2_max, float(output_relative_l2))
            actual_output_cosine_min = min(actual_output_cosine_min, float(output_cosine))
            actual_output_max_absolute_error = max(
                actual_output_max_absolute_error, float(output_diff.abs().max())
            )
            entropy_min = min(entropy_min, float(entropy.min()))
            entropy_max = max(entropy_max, float(entropy.max()))
            finite_ok = (
                finite_ok
                and bool(torch.isfinite(masses).all())
                and bool(torch.isfinite(value_magnitude).all())
                and bool(torch.isfinite(value_share).all())
                and bool(torch.isfinite(group_output).all())
                and bool(torch.isfinite(cancellation_ratio).all())
                and bool(torch.isfinite(direction_contribution).all())
                and bool(torch.isfinite(entropy).all())
            )

            if latent == 1 and start == 0:
                # Independent one-query/one-head dense reference.
                reference_logits = torch.mv(k_heads[0], q_heads[0, 0]) * float(scaling)
                reference_probs = torch.softmax(reference_logits, dim=0)
                reference_masses = torch.stack(
                    [reference_probs.index_select(0, indexes).sum() for indexes in group_indexes]
                )
                reference_entropy = -(reference_probs * torch.log(reference_probs.clamp_min(eps))).sum() / log_total_keys
                reference_weighted = reference_probs * value_l2[0]
                reference_value_share = torch.stack(
                    [reference_weighted.index_select(0, indexes).sum() for indexes in group_indexes]
                ) / reference_weighted.sum().clamp_min(eps)
                dense_reference_error_max = max(
                    float((reference_masses - masses[0, 0]).abs().max()),
                    float((reference_entropy - entropy[0, 0]).abs()),
                    float((reference_value_share - value_share[0, 0]).abs().max()),
                )

        masses_all = torch.cat(masses_chunks, dim=0)  # [query_token,q_head,key_group]
        value_magnitude_all = torch.cat(value_magnitude_chunks, dim=0)
        value_share_all = torch.cat(value_share_chunks, dim=0)
        group_output_norm_all = torch.cat(group_output_norm_chunks, dim=0)
        cancellation_ratio_all = torch.cat(cancellation_ratio_chunks, dim=0)
        direction_contribution_all = torch.cat(direction_contribution_chunks, dim=0)
        entropy_all = torch.cat(entropy_chunks, dim=0)  # [query_token,q_head]
        for head in range(num_q_heads):
            entropy_rows.append(
                {
                    "query_latent": f"L{latent}",
                    "head": head,
                    "num_query_tokens": len(query_positions),
                    "total_key_tokens": total_keys,
                    **_prefix_summary("entropy", entropy_all[:, head], torch),
                }
            )
            for group_slot, (group_name, positions) in enumerate(group_positions):
                baseline = len(positions) / total_keys
                mass = masses_all[:, head, group_slot]
                value_magnitude_group = value_magnitude_all[:, head, group_slot]
                value_share_group = value_share_all[:, head, group_slot]
                group_output_norm_group = group_output_norm_all[:, head, group_slot]
                cancellation_ratio_group = cancellation_ratio_all[:, head, group_slot]
                direction_contribution_group = direction_contribution_all[:, head, group_slot]
                capacity_fraction = float(value_capacity_fraction[head, group_slot])
                value_enrichment = value_share_group / (capacity_fraction + eps)
                value_reweight_ratio = value_share_group / (mass + eps)
                enrichment = mass / (baseline + eps)
                log2_enrichment = torch.log2(enrichment.clamp_min(eps))
                enrichment_identity_error_max = max(
                    enrichment_identity_error_max,
                    float((enrichment * baseline - mass).abs().max()),
                )
                value_enrichment_identity_error_max = max(
                    value_enrichment_identity_error_max,
                    float((value_enrichment * capacity_fraction - value_share_group).abs().max()),
                )
                value_reweight_identity_error_max = max(
                    value_reweight_identity_error_max,
                    float((value_reweight_ratio * mass - value_share_group).abs().max()),
                )
                group_rows.append(
                    {
                        "query_latent": f"L{latent}",
                        "head": head,
                        "key_group": group_name,
                        "num_query_tokens": len(query_positions),
                        "num_key_tokens": len(positions),
                        "total_key_tokens": total_keys,
                        "baseline_fraction": baseline,
                        "value_capacity_fraction": capacity_fraction,
                        **_prefix_summary("mass", mass, torch),
                        **_prefix_summary("enrichment", enrichment, torch),
                        **_prefix_summary("log2_enrichment", log2_enrichment, torch),
                        **_prefix_summary("value_weighted_magnitude", value_magnitude_group, torch),
                        **_prefix_summary("value_weighted_share", value_share_group, torch),
                        **_prefix_summary("value_weighted_enrichment", value_enrichment, torch),
                        **_prefix_summary("value_reweight_ratio", value_reweight_ratio, torch),
                        **_prefix_summary("group_output_norm", group_output_norm_group, torch),
                        **_prefix_summary("cancellation_ratio", cancellation_ratio_group, torch),
                        **_prefix_summary("direction_contribution", direction_contribution_group, torch),
                    }
                )

    validation = {
        "group_partition_ok": partition_ok,
        "mass_sum_error_max": mass_sum_error_max,
        "enrichment_identity_error_max": enrichment_identity_error_max,
        "entropy_min": entropy_min,
        "entropy_max": entropy_max,
        "finite_ok": finite_ok,
        "dense_reference_error_max": dense_reference_error_max,
        "gqa_repeat_factor": gqa_repeat_factor,
        "value_share_sum_error_max": value_share_sum_error_max,
        "value_enrichment_identity_error_max": value_enrichment_identity_error_max,
        "value_reweight_identity_error_max": value_reweight_identity_error_max,
        "alpha_sum_error_max": alpha_sum_error_max,
        "cancellation_ratio_min": cancellation_ratio_min,
        "cancellation_ratio_max": cancellation_ratio_max,
        "actual_output_relative_l2_max": actual_output_relative_l2_max,
        "actual_output_cosine_min": actual_output_cosine_min,
        "actual_output_max_absolute_error": actual_output_max_absolute_error,
    }
    return group_rows, entropy_rows, value_rows, validation


class GenAttentionStatsCollector:
    """Install request-local hooks and stream grouped attention summaries to CSV."""

    def __init__(
        self,
        *,
        torch: Any,
        net: Any,
        output_dir: Path,
        guidance: float,
        num_steps: int,
        shift: float,
        task: str,
        chunk: int,
        selected_steps: Sequence[int] | None = None,
        selected_blocks: Sequence[int] | None = None,
        selected_branches: Sequence[str] = ("conditional", "unconditional"),
        query_chunk_size: int = 64,
    ) -> None:
        self.torch = torch
        self.net = net
        self.output_dir = Path(output_dir)
        self.guidance = float(guidance)
        self.num_steps = int(num_steps)
        self.shift = float(shift)
        self.task = task
        self.chunk = int(chunk)
        self.selected_steps = tuple(range(num_steps) if selected_steps is None else sorted(map(int, selected_steps)))
        self.selected_branches = tuple(selected_branches)
        self.query_chunk_size = int(query_chunk_size)

        layers = getattr(getattr(getattr(net, "language_model", None), "model", None), "layers", None)
        if layers is None:
            raise RuntimeError("Could not locate net.language_model.model.layers")
        self.layers = list(layers)
        self.selected_blocks = tuple(
            range(len(self.layers)) if selected_blocks is None else sorted(map(int, selected_blocks))
        )
        invalid = [block for block in self.selected_blocks if block < 0 or block >= len(self.layers)]
        if invalid:
            raise ValueError(f"Invalid Transformer blocks for {len(self.layers)} layers: {invalid}")
        if self.guidance == 1.0 and "unconditional" in self.selected_branches:
            raise ValueError("Unconditional statistics require CFG guidance != 1")

        self._handles: list[Any] = []
        self._attention_modules: list[Any] = []
        self._current: dict[str, Any] | None = None
        self._forward_call_index = 0
        self._written: set[tuple[int, str, int]] = set()
        self._validation: list[dict[str, Any]] = []
        self._timesteps: dict[tuple[int, str], float] = {}
        self._token_layout: dict[str, Any] | None = None
        self._files: list[Any] = []
        self._group_writer: Any | None = None
        self._entropy_writer: Any | None = None
        self._value_writer: Any | None = None
        self._call_writer: Any | None = None

    def __enter__(self) -> "GenAttentionStatsCollector":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for path in (
            self.output_dir / "group_attention_stats.csv",
            self.output_dir / "entropy_stats.csv",
            self.output_dir / "value_scale_stats.csv",
            self.output_dir / "calls.csv",
            self.output_dir / "validation.json",
        ):
            if path.exists():
                raise FileExistsError(f"Attention statistics output already exists: {path}")
        group_file = (self.output_dir / "group_attention_stats.csv").open("w", newline="", encoding="utf-8")
        entropy_file = (self.output_dir / "entropy_stats.csv").open("w", newline="", encoding="utf-8")
        value_file = (self.output_dir / "value_scale_stats.csv").open("w", newline="", encoding="utf-8")
        call_file = (self.output_dir / "calls.csv").open("w", newline="", encoding="utf-8")
        self._files = [group_file, entropy_file, value_file, call_file]
        self._group_writer = csv.DictWriter(group_file, fieldnames=GROUP_FIELDS)
        self._entropy_writer = csv.DictWriter(entropy_file, fieldnames=ENTROPY_FIELDS)
        self._value_writer = csv.DictWriter(value_file, fieldnames=VALUE_FIELDS)
        self._call_writer = csv.DictWriter(call_file, fieldnames=CALL_FIELDS)
        self._group_writer.writeheader()
        self._entropy_writer.writeheader()
        self._value_writer.writeheader()
        self._call_writer.writeheader()

        self._handles.append(self.net.register_forward_pre_hook(self._network_pre_hook, with_kwargs=True))
        self._handles.append(self.net.register_forward_hook(self._network_post_hook, with_kwargs=True))
        for block in self.selected_blocks:
            attention = getattr(self.layers[block], "self_attn", None)
            if attention is None or not hasattr(attention, "_attention_stats_capture_callback"):
                raise RuntimeError(f"Transformer block {block} does not expose attention statistics hook")
            if attention._attention_stats_capture_callback is not None:
                raise RuntimeError(f"Transformer block {block} already has an attention statistics callback")
            attention._attention_stats_capture_callback = self._capture_attention
            self._attention_modules.append(attention)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        for attention in self._attention_modules:
            attention._attention_stats_capture_callback = None
        self._attention_modules.clear()
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._current = None
        for handle in self._files:
            handle.close()
        self._files.clear()

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
        call_index = self._forward_call_index
        self._forward_call_index += 1
        step, branch = self._call_semantics(call_index)
        if step not in self.selected_steps or branch not in self.selected_branches:
            self._current = None
            return
        layout = _gen_token_layout(self.torch, packed_sequence)
        if self._token_layout is None:
            self._token_layout = layout
        elif layout != self._token_layout:
            raise RuntimeError("GEN token layout changed across selected calls")
        timesteps = getattr(getattr(packed_sequence, "vision", None), "timesteps", None)
        timestep = float("nan")
        if self.torch.is_tensor(timesteps) and timesteps.numel():
            timestep = float(timesteps.reshape(-1)[0].detach().float().cpu())
        self._timesteps[(step, branch)] = timestep
        self._current = {"step": step, "branch": branch, "timestep": timestep}

    def _network_post_hook(
        self, module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any], output: Any
    ) -> None:
        del module, args, kwargs, output
        self._current = None

    def _base_row(self, *, block: int) -> dict[str, Any]:
        assert self._current is not None
        return {
            "shift": self.shift,
            "task": self.task,
            "chunk": self.chunk,
            "branch": self._current["branch"],
            "step": self._current["step"],
            "timestep": self._current["timestep"],
            "block": block,
        }

    def _capture_attention(
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
        current = self._current
        if current is None or layer_index not in self.selected_blocks:
            return
        key = (int(current["step"]), str(current["branch"]), int(layer_index))
        if key in self._written:
            raise RuntimeError(f"Attention statistics saw call/block twice: {key}")
        if self._token_layout is None:
            raise RuntimeError("Missing GEN token layout")
        group_rows, entropy_rows, value_rows, validation = compute_group_attention_statistics(
            torch=self.torch,
            q_gen=q_gen,
            k_ar=k_ar,
            k_gen=k_gen,
            v_ar=v_ar,
            v_gen=v_gen,
            attn_output_gen=attn_output_gen,
            scaling=scaling,
            token_layout=self._token_layout,
            query_chunk_size=self.query_chunk_size,
        )
        base = self._base_row(block=layer_index)
        assert (
            self._group_writer is not None
            and self._entropy_writer is not None
            and self._value_writer is not None
            and self._call_writer is not None
        )
        for row in group_rows:
            self._group_writer.writerow({**base, **row})
        for row in entropy_rows:
            self._entropy_writer.writerow({**base, **row})
        for row in value_rows:
            self._value_writer.writerow({**base, **row})
        self._call_writer.writerow(
            {
                **base,
                "num_ar_tokens": int(k_ar.shape[0]),
                "num_gen_tokens": int(k_gen.shape[0]),
                "total_key_tokens": int(k_ar.shape[0] + k_gen.shape[0]),
                "num_q_heads": int(q_gen.shape[1]),
                "num_kv_heads": int(k_gen.shape[1]),
                "gqa_repeat_factor": int(q_gen.shape[1] // k_gen.shape[1]),
                "head_dim": int(q_gen.shape[2]),
            }
        )
        self._validation.append({**base, **validation})
        self._written.add(key)

    def finish(self) -> dict[str, Any]:
        expected = len(self.selected_steps) * len(self.selected_branches) * len(self.selected_blocks)
        if len(self._written) != expected:
            raise RuntimeError(f"Incomplete attention statistics: call-blocks={len(self._written)}/{expected}")
        if self._token_layout is None:
            raise RuntimeError("Attention statistics captured no token layout")
        summary = {
            "schema_version": 3,
            "capture_type": "group_attention_true_output_and_value_aware",
            "shift": self.shift,
            "task": self.task,
            "chunk": self.chunk,
            "guidance": self.guidance,
            "num_steps": self.num_steps,
            "selected_steps": list(self.selected_steps),
            "selected_blocks": list(self.selected_blocks),
            "selected_branches": list(self.selected_branches),
            "query_latents": [f"L{value}" for value in range(1, 9)],
            "key_groups": ["K_AR", *[f"K_L{value}" for value in range(9)], "K_action"],
            "token_layout": self._token_layout,
            "timesteps": [
                {"step": step, "branch": branch, "timestep": timestep}
                for (step, branch), timestep in sorted(self._timesteps.items())
            ],
            "validation": {
                "group_partition_ok": all(row["group_partition_ok"] for row in self._validation),
                "finite_ok": all(row["finite_ok"] for row in self._validation),
                "mass_sum_error_max": max(row["mass_sum_error_max"] for row in self._validation),
                "enrichment_identity_error_max": max(
                    row["enrichment_identity_error_max"] for row in self._validation
                ),
                "dense_reference_error_max": max(row["dense_reference_error_max"] for row in self._validation),
                "gqa_repeat_factor": int(self._validation[0]["gqa_repeat_factor"]),
                "value_share_sum_error_max": max(
                    row["value_share_sum_error_max"] for row in self._validation
                ),
                "value_enrichment_identity_error_max": max(
                    row["value_enrichment_identity_error_max"] for row in self._validation
                ),
                "value_reweight_identity_error_max": max(
                    row["value_reweight_identity_error_max"] for row in self._validation
                ),
                "alpha_sum_error_max": max(row["alpha_sum_error_max"] for row in self._validation),
                "cancellation_ratio_min": min(
                    row["cancellation_ratio_min"] for row in self._validation
                ),
                "cancellation_ratio_max": max(
                    row["cancellation_ratio_max"] for row in self._validation
                ),
                "actual_output_relative_l2_max": max(
                    row["actual_output_relative_l2_max"] for row in self._validation
                ),
                "actual_output_cosine_min": min(
                    row["actual_output_cosine_min"] for row in self._validation
                ),
                "actual_output_max_absolute_error": max(
                    row["actual_output_max_absolute_error"] for row in self._validation
                ),
                "entropy_min": min(row["entropy_min"] for row in self._validation),
                "entropy_max": max(row["entropy_max"] for row in self._validation),
                "call_block_count": len(self._validation),
            },
        }
        (self.output_dir / "validation.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary
