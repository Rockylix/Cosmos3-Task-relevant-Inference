# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Read-only Action-Query grouped attention/output statistics for RoboLab."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from cosmos_framework.scripts.robolab_attention_stats_capture import _gen_token_layout

GROUP_FIELDS = [
    "shift",
    "task",
    "chunk",
    "branch",
    "step",
    "timestep",
    "block",
    "query_action_index",
    "query_role",
    "action_horizon",
    "head",
    "kv_head",
    "key_group",
    "num_key_tokens",
    "total_key_tokens",
    "baseline_fraction",
    "value_capacity_fraction",
    "mass",
    "enrichment",
    "group_output_norm",
    "cancellation_ratio",
    "direction_contribution",
]
ENTROPY_FIELDS = [
    "shift",
    "task",
    "chunk",
    "branch",
    "step",
    "timestep",
    "block",
    "query_action_index",
    "query_role",
    "action_horizon",
    "head",
    "kv_head",
    "total_key_tokens",
    "normalized_entropy",
]
CALL_FIELDS = [
    "shift",
    "task",
    "chunk",
    "branch",
    "step",
    "timestep",
    "block",
    "num_action_queries",
    "num_predicted_action_queries",
    "num_condition_action_queries",
    "num_ar_tokens",
    "num_gen_tokens",
    "total_key_tokens",
    "num_q_heads",
    "num_kv_heads",
    "gqa_repeat_factor",
    "group_partition_ok",
    "mass_sum_error_max",
    "alpha_sum_error_max",
    "actual_output_relative_l2",
    "actual_output_cosine",
    "actual_output_max_absolute_error",
    "entropy_min",
    "entropy_max",
    "cancellation_ratio_min",
    "cancellation_ratio_max",
    "finite",
]


def _action_query_layout(torch: Any, packed_sequence: Any) -> dict[str, Any]:
    layout = _gen_token_layout(torch, packed_sequence)
    action = getattr(packed_sequence, "action", None)
    if action is None:
        raise RuntimeError("Action Query statistics require packed action tokens")
    action_positions = list(map(int, layout["action_positions"]))
    token_shapes = getattr(action, "token_shapes", None)
    if not token_shapes or len(token_shapes) != 1:
        raise RuntimeError(f"Expected one action token shape, got {token_shapes}")
    token_shape = tuple(int(value) for value in token_shapes[0])
    if math.prod(token_shape) != len(action_positions):
        raise RuntimeError(
            f"Action token shape {token_shape} does not match {len(action_positions)} positions"
        )
    condition_mask = getattr(action, "condition_mask", None)
    if isinstance(condition_mask, (list, tuple)):
        if len(condition_mask) != 1:
            raise RuntimeError(
                f"Expected one sample's action condition mask, got {len(condition_mask)} entries"
            )
        condition_mask = condition_mask[0]
    if not torch.is_tensor(condition_mask) or int(condition_mask.numel()) != len(action_positions):
        condition_shape = tuple(condition_mask.shape) if torch.is_tensor(condition_mask) else None
        raise RuntimeError(
            f"Expected an action condition mask with {len(action_positions)} entries, got "
            f"{condition_shape} ({type(condition_mask).__name__})"
        )
    condition_flags = [bool(value) for value in condition_mask.detach().reshape(-1).cpu().tolist()]
    predicted_horizon = 0
    queries = []
    for action_index, (gen_position, is_condition) in enumerate(
        zip(action_positions, condition_flags, strict=True)
    ):
        horizon = -1 if is_condition else predicted_horizon
        if not is_condition:
            predicted_horizon += 1
        queries.append(
            {
                "query_action_index": action_index,
                "gen_position": gen_position,
                "query_role": "condition" if is_condition else "predicted",
                "action_horizon": horizon,
            }
        )
    layout["action_token_shape"] = list(token_shape)
    layout["action_queries"] = queries
    return layout


def compute_action_query_statistics(
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
    eps: float = 1e-12,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Compute per Action Query token/head/K-group attention and real output."""

    tensors = {
        "q_gen": q_gen,
        "k_ar": k_ar,
        "k_gen": k_gen,
        "v_ar": v_ar,
        "v_gen": v_gen,
        "attn_output_gen": attn_output_gen,
    }
    if any(tensor.ndim != 3 for tensor in tensors.values()):
        raise RuntimeError("Q/K/V/O must all be [tokens,heads,head_dim]")
    num_gen_tokens = int(token_layout["num_gen_tokens"])
    if any(int(tensor.shape[0]) != num_gen_tokens for tensor in (q_gen, k_gen, v_gen, attn_output_gen)):
        raise RuntimeError("GEN Q/K/V/O token count does not match the live layout")
    if k_ar.shape != v_ar.shape or k_gen.shape != v_gen.shape:
        raise RuntimeError("K/V geometry differs")
    if q_gen.shape != attn_output_gen.shape:
        raise RuntimeError("Q and actual attention output geometry differs")
    if q_gen.shape[2] != k_gen.shape[2] or q_gen.shape[2] != k_ar.shape[2]:
        raise RuntimeError("Q/K head dimensions differ")
    num_q_heads = int(q_gen.shape[1])
    num_kv_heads = int(k_gen.shape[1])
    if int(k_ar.shape[1]) != num_kv_heads or num_q_heads % num_kv_heads != 0:
        raise RuntimeError(f"Invalid GQA geometry: q_heads={num_q_heads}, kv_heads={num_kv_heads}")
    gqa_repeat_factor = num_q_heads // num_kv_heads

    k_all = torch.cat([k_ar, k_gen], dim=0).detach().float()
    v_all = torch.cat([v_ar, v_gen], dim=0).detach().float()
    k_heads = k_all.repeat_interleave(gqa_repeat_factor, dim=1).permute(1, 0, 2).contiguous()
    v_heads = v_all.repeat_interleave(gqa_repeat_factor, dim=1).permute(1, 0, 2).contiguous()
    value_l2 = torch.linalg.vector_norm(v_heads, dim=-1)
    total_keys = int(k_all.shape[0])
    num_ar_tokens = int(k_ar.shape[0])

    latent_positions = token_layout["latent_positions"]
    action_positions = list(map(int, token_layout["action_positions"]))
    group_positions: list[tuple[str, list[int]]] = [("K_AR", list(range(num_ar_tokens)))]
    for latent in range(9):
        group_positions.append(
            (
                f"K_L{latent}",
                [num_ar_tokens + int(position) for position in latent_positions[f"L{latent}"]],
            )
        )
    group_positions.append(
        ("K_action", [num_ar_tokens + position for position in action_positions])
    )
    flat_positions = [position for _name, positions in group_positions for position in positions]
    partition_ok = len(flat_positions) == total_keys and sorted(flat_positions) == list(range(total_keys))
    if not partition_ok:
        raise RuntimeError("K_AR, K_L0..K_L8 and K_action do not partition the actual keys")
    group_indexes = [
        torch.tensor(positions, dtype=torch.long, device=q_gen.device)
        for _name, positions in group_positions
    ]
    value_capacity = torch.stack(
        [value_l2.index_select(-1, index).sum(-1) for index in group_indexes], dim=-1
    )
    value_capacity_fraction = value_capacity / value_l2.sum(-1, keepdim=True).clamp_min(eps)

    action_queries = list(token_layout["action_queries"])
    query_index = torch.tensor(
        [int(query["gen_position"]) for query in action_queries],
        dtype=torch.long,
        device=q_gen.device,
    )
    q_heads = q_gen.index_select(0, query_index).detach().float().permute(1, 0, 2).contiguous()
    logits = torch.bmm(q_heads, k_heads.transpose(1, 2)) * float(scaling)
    probabilities = torch.softmax(logits, dim=-1)
    masses = torch.stack(
        [probabilities.index_select(-1, index).sum(-1) for index in group_indexes], dim=-1
    )
    weighted_value_norm = probabilities * value_l2[:, None, :]
    value_magnitude = torch.stack(
        [weighted_value_norm.index_select(-1, index).sum(-1) for index in group_indexes],
        dim=-1,
    )
    group_output = torch.stack(
        [
            torch.bmm(
                probabilities.index_select(-1, index),
                v_heads.index_select(1, index),
            )
            for index in group_indexes
        ],
        dim=2,
    )  # [q_head,action_query,key_group,head_dim]
    total_output = group_output.sum(dim=2)
    group_output_norm = torch.linalg.vector_norm(group_output, dim=-1)
    cancellation_ratio = group_output_norm / value_magnitude.clamp_min(eps)
    total_output_energy = total_output.square().sum(dim=-1, keepdim=True)
    direction_contribution = (
        (group_output * total_output.unsqueeze(2)).sum(dim=-1)
        / total_output_energy.clamp_min(eps)
    )
    log_total_keys = math.log(total_keys)
    if log_total_keys <= 0:
        raise RuntimeError("Normalized entropy requires at least two keys")
    normalized_entropy = -(
        probabilities * torch.log(probabilities.clamp_min(eps))
    ).sum(-1) / log_total_keys

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

    group_rows: list[dict[str, Any]] = []
    entropy_rows: list[dict[str, Any]] = []
    masses_cpu = masses.detach().cpu()
    output_norm_cpu = group_output_norm.detach().cpu()
    cancellation_cpu = cancellation_ratio.detach().cpu()
    direction_cpu = direction_contribution.detach().cpu()
    entropy_cpu = normalized_entropy.detach().cpu()
    capacity_cpu = value_capacity_fraction.detach().cpu()
    for query_slot, query in enumerate(action_queries):
        query_metadata = {
            "query_action_index": int(query["query_action_index"]),
            "query_role": str(query["query_role"]),
            "action_horizon": int(query["action_horizon"]),
        }
        for head in range(num_q_heads):
            kv_head = head // gqa_repeat_factor
            entropy_rows.append(
                {
                    **query_metadata,
                    "head": head,
                    "kv_head": kv_head,
                    "total_key_tokens": total_keys,
                    "normalized_entropy": float(entropy_cpu[head, query_slot]),
                }
            )
            for group_slot, (group_name, positions) in enumerate(group_positions):
                baseline_fraction = len(positions) / total_keys
                mass = float(masses_cpu[head, query_slot, group_slot])
                group_rows.append(
                    {
                        **query_metadata,
                        "head": head,
                        "kv_head": kv_head,
                        "key_group": group_name,
                        "num_key_tokens": len(positions),
                        "total_key_tokens": total_keys,
                        "baseline_fraction": baseline_fraction,
                        "value_capacity_fraction": float(capacity_cpu[head, group_slot]),
                        "mass": mass,
                        "enrichment": mass / (baseline_fraction + eps),
                        "group_output_norm": float(output_norm_cpu[head, query_slot, group_slot]),
                        "cancellation_ratio": float(cancellation_cpu[head, query_slot, group_slot]),
                        "direction_contribution": float(direction_cpu[head, query_slot, group_slot]),
                    }
                )

    finite = all(
        bool(torch.isfinite(value).all())
        for value in (
            probabilities,
            masses,
            group_output,
            group_output_norm,
            cancellation_ratio,
            direction_contribution,
            normalized_entropy,
        )
    )
    validation = {
        "group_partition_ok": partition_ok,
        "finite": finite,
        "num_action_queries": len(action_queries),
        "num_predicted_action_queries": sum(query["query_role"] == "predicted" for query in action_queries),
        "num_condition_action_queries": sum(query["query_role"] == "condition" for query in action_queries),
        "num_ar_tokens": num_ar_tokens,
        "num_gen_tokens": num_gen_tokens,
        "total_key_tokens": total_keys,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "gqa_repeat_factor": gqa_repeat_factor,
        "mass_sum_error_max": float((masses.sum(-1) - 1.0).abs().max()),
        "alpha_sum_error_max": float((direction_contribution.sum(-1) - 1.0).abs().max()),
        "actual_output_relative_l2": float(output_relative_l2),
        "actual_output_cosine": float(output_cosine),
        "actual_output_max_absolute_error": float(output_diff.abs().max()),
        "entropy_min": float(normalized_entropy.min()),
        "entropy_max": float(normalized_entropy.max()),
        "cancellation_ratio_min": float(cancellation_ratio.min()),
        "cancellation_ratio_max": float(cancellation_ratio.max()),
    }
    return group_rows, entropy_rows, validation


class ActionQueryAttentionCollector:
    """Install request-local Action Query capture callbacks and stream CSVs."""

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
        selected_blocks: Sequence[int] = (0, 4, 8, 12, 16, 20, 24),
        selected_branches: Sequence[str] = ("conditional", "unconditional"),
    ) -> None:
        self.torch = torch
        self.net = net
        self.output_dir = Path(output_dir)
        self.guidance = float(guidance)
        self.num_steps = int(num_steps)
        self.shift = float(shift)
        self.task = str(task)
        self.chunk = int(chunk)
        self.selected_steps = tuple(
            range(num_steps) if selected_steps is None else sorted(map(int, selected_steps))
        )
        self.selected_blocks = tuple(sorted(map(int, selected_blocks)))
        self.selected_branches = tuple(selected_branches)
        layers = getattr(getattr(getattr(net, "language_model", None), "model", None), "layers", None)
        if layers is None:
            raise RuntimeError("Could not locate net.language_model.model.layers")
        self.layers = list(layers)
        invalid = [block for block in self.selected_blocks if block < 0 or block >= len(self.layers)]
        if invalid:
            raise ValueError(f"Invalid Transformer blocks for {len(self.layers)} layers: {invalid}")
        if self.guidance == 1.0 and "unconditional" in self.selected_branches:
            raise ValueError("Unconditional capture requires CFG guidance != 1")

        self._handles: list[Any] = []
        self._attention_modules: list[Any] = []
        self._files: list[Any] = []
        self._group_writer: Any | None = None
        self._entropy_writer: Any | None = None
        self._call_writer: Any | None = None
        self._current: dict[str, Any] | None = None
        self._forward_call_index = 0
        self._token_layout: dict[str, Any] | None = None
        self._written: set[tuple[int, str, int]] = set()
        self._timesteps: dict[tuple[int, str], float] = {}
        self._validation: list[dict[str, Any]] = []

    def __enter__(self) -> "ActionQueryAttentionCollector":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "group": self.output_dir / "action_query_group_metrics.csv",
            "entropy": self.output_dir / "action_query_entropy.csv",
            "calls": self.output_dir / "calls.csv",
        }
        for path in (*paths.values(), self.output_dir / "validation.json"):
            if path.exists():
                raise FileExistsError(f"Action Query output already exists: {path}")
        group_file = paths["group"].open("w", newline="", encoding="utf-8")
        entropy_file = paths["entropy"].open("w", newline="", encoding="utf-8")
        call_file = paths["calls"].open("w", newline="", encoding="utf-8")
        self._files = [group_file, entropy_file, call_file]
        self._group_writer = csv.DictWriter(group_file, fieldnames=GROUP_FIELDS)
        self._entropy_writer = csv.DictWriter(entropy_file, fieldnames=ENTROPY_FIELDS)
        self._call_writer = csv.DictWriter(call_file, fieldnames=CALL_FIELDS)
        self._group_writer.writeheader()
        self._entropy_writer.writeheader()
        self._call_writer.writeheader()
        self._handles.append(self.net.register_forward_pre_hook(self._network_pre_hook, with_kwargs=True))
        self._handles.append(self.net.register_forward_hook(self._network_post_hook, with_kwargs=True))
        for block in self.selected_blocks:
            attention = getattr(self.layers[block], "self_attn", None)
            if attention is None or not hasattr(attention, "_attention_stats_capture_callback"):
                raise RuntimeError(f"Transformer block {block} does not expose attention capture hook")
            if attention._attention_stats_capture_callback is not None:
                raise RuntimeError(f"Transformer block {block} already has an attention callback")
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
        for handle in self._files:
            handle.close()
        self._files.clear()
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
        call_index = self._forward_call_index
        self._forward_call_index += 1
        step, branch = self._call_semantics(call_index)
        if step not in self.selected_steps or branch not in self.selected_branches:
            self._current = None
            return
        layout = _action_query_layout(self.torch, packed_sequence)
        if self._token_layout is None:
            self._token_layout = layout
        elif layout != self._token_layout:
            raise RuntimeError("GEN/action token layout changed across selected calls")
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

    def _base_row(self, block: int) -> dict[str, Any]:
        assert self._current is not None
        return {
            "shift": self.shift,
            "task": self.task,
            "chunk": self.chunk,
            "branch": self._current["branch"],
            "step": self._current["step"],
            "timestep": self._current["timestep"],
            "block": int(block),
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
        if self._current is None or layer_index not in self.selected_blocks:
            return
        key = (int(self._current["step"]), str(self._current["branch"]), int(layer_index))
        if key in self._written:
            raise RuntimeError(f"Action Query call/block captured twice: {key}")
        if self._token_layout is None:
            raise RuntimeError("Missing Action Query token layout")
        group_rows, entropy_rows, validation = compute_action_query_statistics(
            torch=self.torch,
            q_gen=q_gen,
            k_ar=k_ar,
            k_gen=k_gen,
            v_ar=v_ar,
            v_gen=v_gen,
            attn_output_gen=attn_output_gen,
            scaling=scaling,
            token_layout=self._token_layout,
        )
        base = self._base_row(layer_index)
        assert self._group_writer is not None
        assert self._entropy_writer is not None
        assert self._call_writer is not None
        for row in group_rows:
            self._group_writer.writerow({**base, **row})
        for row in entropy_rows:
            self._entropy_writer.writerow({**base, **row})
        self._call_writer.writerow({**base, **validation})
        self._validation.append({**base, **validation})
        self._written.add(key)

    def finish(self) -> dict[str, Any]:
        expected = len(self.selected_steps) * len(self.selected_branches) * len(self.selected_blocks)
        if len(self._written) != expected:
            raise RuntimeError(f"Incomplete Action Query capture: {len(self._written)}/{expected} call-blocks")
        if self._token_layout is None or not self._validation:
            raise RuntimeError("Action Query capture produced no data")
        profile = {
            "schema_version": 1,
            "capture_type": "action_query_group_attention_and_true_output",
            "shift": self.shift,
            "task": self.task,
            "chunk": self.chunk,
            "guidance": self.guidance,
            "num_steps": self.num_steps,
            "selected_steps": list(self.selected_steps),
            "selected_blocks": list(self.selected_blocks),
            "selected_branches": list(self.selected_branches),
            "key_groups": ["K_AR", *[f"K_L{latent}" for latent in range(9)], "K_action"],
            "token_layout": self._token_layout,
            "timesteps": [
                {"step": step, "branch": branch, "timestep": timestep}
                for (step, branch), timestep in sorted(self._timesteps.items())
            ],
            "validation": {
                "complete": True,
                "call_block_count": len(self._validation),
                "group_partition_ok": all(row["group_partition_ok"] for row in self._validation),
                "finite": all(row["finite"] for row in self._validation),
                "num_action_queries": int(self._validation[0]["num_action_queries"]),
                "num_predicted_action_queries": int(
                    self._validation[0]["num_predicted_action_queries"]
                ),
                "num_condition_action_queries": int(
                    self._validation[0]["num_condition_action_queries"]
                ),
                "gqa_repeat_factor": int(self._validation[0]["gqa_repeat_factor"]),
                "mass_sum_error_max": max(row["mass_sum_error_max"] for row in self._validation),
                "alpha_sum_error_max": max(row["alpha_sum_error_max"] for row in self._validation),
                "actual_output_relative_l2_max": max(
                    row["actual_output_relative_l2"] for row in self._validation
                ),
                "actual_output_cosine_min": min(
                    row["actual_output_cosine"] for row in self._validation
                ),
                "actual_output_max_absolute_error": max(
                    row["actual_output_max_absolute_error"] for row in self._validation
                ),
                "entropy_min": min(row["entropy_min"] for row in self._validation),
                "entropy_max": max(row["entropy_max"] for row in self._validation),
                "cancellation_ratio_min": min(
                    row["cancellation_ratio_min"] for row in self._validation
                ),
                "cancellation_ratio_max": max(
                    row["cancellation_ratio_max"] for row in self._validation
                ),
            },
            "boundaries": {
                "Q": "post QK-norm and post RoPE Action Query",
                "K": "actual AR cache plus post QK-norm/post RoPE GEN K",
                "V": "actual AR cache plus post projection/head-reshape GEN V",
                "O": "actual attention kernel output before o_proj",
                "GQA": "consecutive Query heads map to KV head by floor(h_q/repeat_factor)",
            },
        }
        (self.output_dir / "validation.json").write_text(
            json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return profile
