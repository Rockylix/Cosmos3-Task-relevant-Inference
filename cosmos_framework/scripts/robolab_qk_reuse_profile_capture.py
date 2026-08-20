# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Read-only future-latent Q/K reuse profiling for Cosmos3 Edge."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cosmos_framework.scripts.robolab_attention_stats_capture import _gen_token_layout

PROFILE_FIELDS = [
    "shift",
    "task",
    "chunk",
    "branch",
    "step",
    "timestep",
    "block",
    "source_latent",
    "target_latent",
    "future_pair",
    "threshold",
] + [
    f"{representation}_{metric}"
    for representation in ("q", "k")
    for metric in (
        "full_cosine",
        "full_relative_l2",
        "norm_ratio",
        "token_cosine_mean",
        "token_cosine_std",
        "token_cosine_p10",
        "token_cosine_p50",
        "token_cosine_p90",
        "token_cosine_min",
        "token_cosine_max",
        "token_fraction_ge_threshold",
    )
] + [
    "joint_full_cosine_min",
    "joint_token_cosine_mean_min",
    "joint_token_cosine_p10_min",
    "candidate_full_cosine",
    "candidate_token_mean",
    "candidate_token_p10",
]


def _representation_metrics(
    *, torch: Any, source: Any, target: Any, threshold: float, eps: float
) -> dict[str, float]:
    source = source.detach().float()
    target = target.detach().float()
    if source.shape != target.shape or source.ndim != 3:
        raise RuntimeError(f"Expected matching [spatial,heads,dim], got {source.shape}/{target.shape}")
    source_flat = source.reshape(-1)
    target_flat = target.reshape(-1)
    full_cosine = torch.nn.functional.cosine_similarity(
        source_flat, target_flat, dim=0, eps=eps
    )
    source_norm = torch.linalg.vector_norm(source_flat)
    target_norm = torch.linalg.vector_norm(target_flat)
    full_relative_l2 = torch.linalg.vector_norm(target_flat - source_flat) / source_norm.clamp_min(eps)
    norm_ratio = target_norm / source_norm.clamp_min(eps)
    source_tokens = source.reshape(source.shape[0], -1)
    target_tokens = target.reshape(target.shape[0], -1)
    token_cosine = torch.nn.functional.cosine_similarity(
        source_tokens, target_tokens, dim=-1, eps=eps
    )
    quantiles = torch.quantile(token_cosine, torch.tensor([0.1, 0.5, 0.9], device=token_cosine.device))
    values = {
        "full_cosine": float(full_cosine),
        "full_relative_l2": float(full_relative_l2),
        "norm_ratio": float(norm_ratio),
        "token_cosine_mean": float(token_cosine.mean()),
        "token_cosine_std": float(token_cosine.std(unbiased=False)),
        "token_cosine_p10": float(quantiles[0]),
        "token_cosine_p50": float(quantiles[1]),
        "token_cosine_p90": float(quantiles[2]),
        "token_cosine_min": float(token_cosine.min()),
        "token_cosine_max": float(token_cosine.max()),
        "token_fraction_ge_threshold": float((token_cosine >= threshold).float().mean()),
    }
    if not all(torch.isfinite(torch.tensor(value)) for value in values.values()):
        raise RuntimeError("Q/K reuse profile contains NaN/Inf")
    return values


def compute_qk_reuse_profile(
    *,
    torch: Any,
    q_gen: Any,
    k_gen: Any,
    token_layout: Mapping[str, Any],
    threshold: float = 0.9,
    eps: float = 1e-12,
) -> list[dict[str, Any]]:
    """Profile all adjacent L1..L8 pairs for post-RoPE Q and K."""

    num_gen_tokens = int(token_layout["num_gen_tokens"])
    if q_gen.ndim != 3 or k_gen.ndim != 3:
        raise RuntimeError("Q/K must be [tokens,heads,head_dim]")
    if int(q_gen.shape[0]) != num_gen_tokens or int(k_gen.shape[0]) != num_gen_tokens:
        raise RuntimeError("Q/K GEN token count does not match live layout")
    if int(q_gen.shape[2]) != int(k_gen.shape[2]):
        raise RuntimeError("Q/K head dimensions differ")
    latent_shape = tuple(map(int, token_layout["latent_shape_thw"]))
    if latent_shape[0] != 9:
        raise RuntimeError(f"Expected L0 plus L1..L8, got {latent_shape}")
    spatial_tokens = latent_shape[1] * latent_shape[2]

    rows = []
    for source_latent in range(1, 8):
        target_latent = source_latent + 1
        source_index = torch.tensor(
            token_layout["latent_positions"][f"L{source_latent}"],
            device=q_gen.device,
            dtype=torch.long,
        )
        target_index = torch.tensor(
            token_layout["latent_positions"][f"L{target_latent}"],
            device=q_gen.device,
            dtype=torch.long,
        )
        if int(source_index.numel()) != spatial_tokens or int(target_index.numel()) != spatial_tokens:
            raise RuntimeError("Future latent spatial-token count changed")
        q_metrics = _representation_metrics(
            torch=torch,
            source=q_gen.index_select(0, source_index),
            target=q_gen.index_select(0, target_index),
            threshold=threshold,
            eps=eps,
        )
        k_metrics = _representation_metrics(
            torch=torch,
            source=k_gen.index_select(0, source_index),
            target=k_gen.index_select(0, target_index),
            threshold=threshold,
            eps=eps,
        )
        joint_full = min(q_metrics["full_cosine"], k_metrics["full_cosine"])
        joint_mean = min(q_metrics["token_cosine_mean"], k_metrics["token_cosine_mean"])
        joint_p10 = min(q_metrics["token_cosine_p10"], k_metrics["token_cosine_p10"])
        rows.append(
            {
                "source_latent": f"L{source_latent}",
                "target_latent": f"L{target_latent}",
                "future_pair": f"L{source_latent}_L{target_latent}",
                "threshold": threshold,
                **{f"q_{key}": value for key, value in q_metrics.items()},
                **{f"k_{key}": value for key, value in k_metrics.items()},
                "joint_full_cosine_min": joint_full,
                "joint_token_cosine_mean_min": joint_mean,
                "joint_token_cosine_p10_min": joint_p10,
                "candidate_full_cosine": joint_full >= threshold,
                "candidate_token_mean": joint_mean >= threshold,
                "candidate_token_p10": joint_p10 >= threshold,
            }
        )
    return rows


class QKReuseProfileCollector:
    """Capture compact Q/K similarity rows from all selected calls and blocks."""

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
        threshold: float = 0.9,
        selected_steps: Sequence[int] | None = None,
        selected_blocks: Sequence[int] | None = None,
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
        self.threshold = float(threshold)
        self.selected_steps = tuple(
            range(num_steps) if selected_steps is None else sorted(map(int, selected_steps))
        )
        self.selected_branches = tuple(selected_branches)
        layers = getattr(getattr(getattr(net, "language_model", None), "model", None), "layers", None)
        if layers is None:
            raise RuntimeError("Could not locate net.language_model.model.layers")
        self.layers = list(layers)
        self.selected_blocks = tuple(
            range(len(self.layers)) if selected_blocks is None else sorted(map(int, selected_blocks))
        )
        invalid = [block for block in self.selected_blocks if block < 0 or block >= len(self.layers)]
        if invalid:
            raise ValueError(f"Invalid blocks for {len(self.layers)} layers: {invalid}")
        if self.guidance == 1.0 and "unconditional" in self.selected_branches:
            raise ValueError("Unconditional profile requires CFG guidance != 1")
        self._handles: list[Any] = []
        self._attention_modules: list[Any] = []
        self._files: list[Any] = []
        self._writer: Any | None = None
        self._current: dict[str, Any] | None = None
        self._forward_call_index = 0
        self._token_layout: dict[str, Any] | None = None
        self._written: set[tuple[int, str, int]] = set()
        self._timesteps: dict[tuple[int, str], float] = {}
        self._candidate_counts = {"full": 0, "mean": 0, "p10": 0}

    def __enter__(self) -> "QKReuseProfileCollector":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / "qk_adjacent_profile.csv"
        validation_path = self.output_dir / "validation.json"
        for path in (csv_path, validation_path):
            if path.exists():
                raise FileExistsError(f"Q/K reuse profile output exists: {path}")
        handle = csv_path.open("w", newline="", encoding="utf-8")
        self._files = [handle]
        self._writer = csv.DictWriter(handle, fieldnames=PROFILE_FIELDS)
        self._writer.writeheader()
        self._handles.append(self.net.register_forward_pre_hook(self._network_pre_hook, with_kwargs=True))
        self._handles.append(self.net.register_forward_hook(self._network_post_hook, with_kwargs=True))
        for block in self.selected_blocks:
            attention = getattr(self.layers[block], "self_attn", None)
            if attention is None or not hasattr(attention, "_attention_stats_capture_callback"):
                raise RuntimeError(f"Block {block} does not expose attention capture hook")
            if attention._attention_stats_capture_callback is not None:
                raise RuntimeError(f"Block {block} already has an attention callback")
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
        layout = _gen_token_layout(self.torch, packed_sequence)
        if self._token_layout is None:
            self._token_layout = layout
        elif layout != self._token_layout:
            raise RuntimeError("GEN token layout changed across calls")
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
        del k_ar, v_ar, v_gen, attn_output_gen, scaling
        if self._current is None or layer_index not in self.selected_blocks:
            return
        key = (int(self._current["step"]), str(self._current["branch"]), int(layer_index))
        if key in self._written:
            raise RuntimeError(f"Q/K profile captured call/block twice: {key}")
        if self._token_layout is None or self._writer is None:
            raise RuntimeError("Q/K profile missing token layout/writer")
        rows = compute_qk_reuse_profile(
            torch=self.torch,
            q_gen=q_gen,
            k_gen=k_gen,
            token_layout=self._token_layout,
            threshold=self.threshold,
        )
        base = {
            "shift": self.shift,
            "task": self.task,
            "chunk": self.chunk,
            "branch": self._current["branch"],
            "step": self._current["step"],
            "timestep": self._current["timestep"],
            "block": int(layer_index),
        }
        for row in rows:
            self._writer.writerow({**base, **row})
            self._candidate_counts["full"] += int(row["candidate_full_cosine"])
            self._candidate_counts["mean"] += int(row["candidate_token_mean"])
            self._candidate_counts["p10"] += int(row["candidate_token_p10"])
        self._written.add(key)

    def finish(self) -> dict[str, Any]:
        expected_call_blocks = (
            len(self.selected_steps) * len(self.selected_branches) * len(self.selected_blocks)
        )
        if len(self._written) != expected_call_blocks or self._token_layout is None:
            raise RuntimeError(
                f"Incomplete Q/K profile: call-blocks={len(self._written)}/{expected_call_blocks}"
            )
        profile = {
            "schema_version": 1,
            "capture_type": "post_rope_future_latent_qk_reuse_profile",
            "shift": self.shift,
            "task": self.task,
            "chunk": self.chunk,
            "guidance": self.guidance,
            "num_steps": self.num_steps,
            "threshold": self.threshold,
            "selected_steps": list(self.selected_steps),
            "selected_blocks": list(self.selected_blocks),
            "selected_branches": list(self.selected_branches),
            "token_layout": self._token_layout,
            "timesteps": [
                {"step": step, "branch": branch, "timestep": timestep}
                for (step, branch), timestep in sorted(self._timesteps.items())
            ],
            "validation": {
                "complete": True,
                "call_block_count": len(self._written),
                "expected_call_block_count": expected_call_blocks,
                "row_count": expected_call_blocks * 7,
                "candidate_counts_at_threshold": self._candidate_counts,
                "finite": True,
            },
            "boundaries": {
                "Q": "post QK-norm and post RoPE, all 16 heads",
                "K": "post QK-norm and post RoPE, all 8 KV heads",
                "pair": "same spatial coordinates in adjacent future latents L1..L8",
            },
        }
        (self.output_dir / "validation.json").write_text(
            json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return profile
