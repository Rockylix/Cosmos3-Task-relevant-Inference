# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Read-only spatial-token similarity capture for Cosmos3 Edge RoboLab runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cosmos_framework.scripts.robolab_attention_stats_capture import _gen_token_layout

CSV_FIELDS = [
    "shift",
    "task",
    "chunk",
    "branch",
    "step",
    "timestep",
    "block",
    "representation",
    "height_index",
    "width_index",
    "adjacent_cosine_mean",
]
REPRESENTATIONS = ("MLP", "Q", "K", "V", "O")


def compute_adjacent_spatial_cosine(
    *,
    torch: Any,
    tensor: Any,
    token_layout: Mapping[str, Any],
    eps: float = 1e-12,
) -> tuple[Any, Any, dict[str, Any]]:
    """Compare L1..L8 at matching spatial coordinates.

    All feature axes after the token axis are flattened.  The first return
    value contains the seven adjacent-pair maps [7,H,W], and the second is
    their arithmetic mean [H,W].
    """

    num_gen_tokens = int(token_layout["num_gen_tokens"])
    if tensor.ndim < 2 or int(tensor.shape[0]) != num_gen_tokens:
        raise RuntimeError(
            f"Expected [num_gen_tokens,...] with N={num_gen_tokens}, got {tuple(tensor.shape)}"
        )
    num_latents, height, width = map(int, token_layout["latent_shape_thw"])
    if num_latents != 9:
        raise RuntimeError(f"Expected L0 plus L1..L8, got {num_latents} latents")
    spatial_tokens = height * width
    features = tensor.detach().float().reshape(num_gen_tokens, -1)
    if int(features.shape[1]) == 0:
        raise RuntimeError("Representation has an empty feature dimension")

    future = []
    for latent in range(1, 9):
        positions = token_layout["latent_positions"][f"L{latent}"]
        if len(positions) != spatial_tokens:
            raise RuntimeError(
                f"L{latent} has {len(positions)} spatial tokens, expected {spatial_tokens}"
            )
        index = torch.as_tensor(positions, device=features.device, dtype=torch.long)
        future.append(features.index_select(0, index))
    future_tensor = torch.stack(future, dim=0)  # [8,H*W,F]
    norms = torch.linalg.vector_norm(future_tensor, dim=-1)
    pair_maps = torch.nn.functional.cosine_similarity(
        future_tensor[:-1], future_tensor[1:], dim=-1, eps=eps
    ).reshape(7, height, width)
    mean_map = pair_maps.mean(dim=0)
    if not bool(torch.isfinite(pair_maps).all()):
        raise RuntimeError("Spatial cosine contains NaN/Inf")
    validation = {
        "input_shape": list(tensor.shape),
        "flattened_feature_dim": int(features.shape[1]),
        "pair_map_shape": list(pair_maps.shape),
        "mean_map_shape": list(mean_map.shape),
        "zero_norm_token_count": int((norms <= eps).sum().detach().cpu()),
        "cosine_min": float(pair_maps.min().detach().cpu()),
        "cosine_max": float(pair_maps.max().detach().cpu()),
        "finite": True,
    }
    return pair_maps.cpu(), mean_map.cpu(), validation


class SpatialTokenSimilarityCollector:
    """Capture MLP/pre-RoPE-QK/V/O maps without changing the forward computation."""

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
        self._rope_qk_modules: list[Any] = []
        self._current: dict[str, Any] | None = None
        self._forward_call_index = 0
        self._token_layout: dict[str, Any] | None = None
        self._timesteps: dict[tuple[int, str], float] = {}
        self._maps: dict[tuple[str, int, str, int], dict[str, Any]] = {}
        self._validation: list[dict[str, Any]] = []

    def __enter__(self) -> "SpatialTokenSimilarityCollector":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("spatial_similarity.csv", "spatial_similarity_maps.pt", "validation.json"):
            path = self.output_dir / name
            if path.exists():
                raise FileExistsError(f"Spatial similarity output already exists: {path}")
        self._handles.append(self.net.register_forward_pre_hook(self._network_pre_hook, with_kwargs=True))
        self._handles.append(self.net.register_forward_hook(self._network_post_hook, with_kwargs=True))
        for block in self.selected_blocks:
            layer = self.layers[block]
            attention = getattr(layer, "self_attn", None)
            if attention is None or not hasattr(attention, "_attention_stats_capture_callback"):
                raise RuntimeError(f"Transformer block {block} does not expose attention capture hook")
            if attention._attention_stats_capture_callback is not None:
                raise RuntimeError(f"Transformer block {block} already has an attention callback")
            if not hasattr(attention, "_rope_qk_capture_callback"):
                raise RuntimeError(f"Transformer block {block} does not expose RoPE Q/K capture hook")
            if attention._rope_qk_capture_callback is not None:
                raise RuntimeError(f"Transformer block {block} already has a RoPE Q/K callback")
            attention._attention_stats_capture_callback = self._capture_attention
            attention._rope_qk_capture_callback = self._capture_pre_rope_qk
            self._attention_modules.append(attention)
            self._rope_qk_modules.append(attention)
            mlp = getattr(layer, "mlp_moe_gen", None)
            if mlp is None:
                raise RuntimeError(f"Transformer block {block} has no generation MLP")
            self._handles.append(
                mlp.register_forward_hook(
                    lambda module, args, output, block=block: self._capture_mlp(block, output)
                )
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        for attention in self._attention_modules:
            attention._attention_stats_capture_callback = None
        self._attention_modules.clear()
        for attention in self._rope_qk_modules:
            attention._rope_qk_capture_callback = None
        self._rope_qk_modules.clear()
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
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

    def _store(self, representation: str, block: int, tensor: Any) -> None:
        if self._current is None:
            return
        if self._token_layout is None:
            raise RuntimeError("Missing GEN token layout")
        key = (
            representation,
            int(self._current["step"]),
            str(self._current["branch"]),
            int(block),
        )
        if key in self._maps:
            raise RuntimeError(f"Spatial representation captured twice: {key}")
        pair_maps, mean_map, validation = compute_adjacent_spatial_cosine(
            torch=self.torch,
            tensor=tensor,
            token_layout=self._token_layout,
        )
        self._maps[key] = {
            "pair_maps": pair_maps,
            "mean_map": mean_map,
            "timestep": float(self._current["timestep"]),
        }
        self._validation.append(
            {
                "representation": representation,
                "step": int(self._current["step"]),
                "branch": str(self._current["branch"]),
                "block": int(block),
                **validation,
            }
        )

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
        del q_gen, k_ar, k_gen, v_ar, scaling
        if self._current is None or layer_index not in self.selected_blocks:
            return
        self._store("V", layer_index, v_gen)
        self._store("O", layer_index, attn_output_gen)

    def _capture_pre_rope_qk(
        self,
        *,
        layer_index: int,
        q_raw: Any,
        k_raw: Any,
        q_rope: Any,
        k_rope: Any,
        cos: Any,
        sin: Any,
    ) -> None:
        """Capture post-QK-norm Q/K immediately before rotary embedding."""

        del q_rope, k_rope, cos, sin
        if self._current is None or layer_index not in self.selected_blocks:
            return
        self._store("Q", layer_index, q_raw)
        self._store("K", layer_index, k_raw)

    def _capture_mlp(self, block: int, output: Any) -> None:
        if self._current is None:
            return
        tensor = output[0] if isinstance(output, tuple) else output
        if not self.torch.is_tensor(tensor):
            raise TypeError(f"Block {block} MLP output is {type(tensor).__name__}, expected Tensor")
        self._store("MLP", block, tensor)

    def finish(self) -> dict[str, Any]:
        expected_keys = {
            (representation, step, branch, block)
            for representation in REPRESENTATIONS
            for step in self.selected_steps
            for branch in self.selected_branches
            for block in self.selected_blocks
        }
        actual_keys = set(self._maps)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise RuntimeError(
                f"Incomplete spatial capture: actual={len(actual_keys)}/{len(expected_keys)}, "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        if self._token_layout is None:
            raise RuntimeError("Spatial capture saw no token layout")

        with (self.output_dir / "spatial_similarity.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for key in sorted(self._maps, key=lambda item: (REPRESENTATIONS.index(item[0]), item[1], item[2], item[3])):
                representation, step, branch, block = key
                item = self._maps[key]
                mean_map = item["mean_map"]
                for height_index in range(int(mean_map.shape[0])):
                    for width_index in range(int(mean_map.shape[1])):
                        writer.writerow(
                            {
                                "shift": self.shift,
                                "task": self.task,
                                "chunk": self.chunk,
                                "branch": branch,
                                "step": step,
                                "timestep": item["timestep"],
                                "block": block,
                                "representation": representation,
                                "height_index": height_index,
                                "width_index": width_index,
                                "adjacent_cosine_mean": float(mean_map[height_index, width_index]),
                            }
                        )
        self.torch.save(self._maps, self.output_dir / "spatial_similarity_maps.pt")
        profile = {
            "schema_version": 2,
            "capture_type": "future_spatial_token_adjacent_cosine",
            "formula": "mean_{f=1..7} cosine(vec(X_f[y,x]), vec(X_{f+1}[y,x]))",
            "shift": self.shift,
            "task": self.task,
            "chunk": self.chunk,
            "guidance": self.guidance,
            "num_steps": self.num_steps,
            "selected_steps": list(self.selected_steps),
            "selected_blocks": list(self.selected_blocks),
            "selected_branches": list(self.selected_branches),
            "representations": list(REPRESENTATIONS),
            "token_layout": self._token_layout,
            "timesteps": [
                {"step": step, "branch": branch, "timestep": timestep}
                for (step, branch), timestep in sorted(self._timesteps.items())
            ],
            "validation": {
                "complete": True,
                "finite": all(row["finite"] for row in self._validation),
                "map_count": len(self._maps),
                "expected_map_count": len(expected_keys),
                "zero_norm_token_count": sum(row["zero_norm_token_count"] for row in self._validation),
                "cosine_min": min(row["cosine_min"] for row in self._validation),
                "cosine_max": max(row["cosine_max"] for row in self._validation),
            },
            "boundaries": {
                "Q": "post Q projection/head reshape/QK-norm and pre RoPE, all 16 query heads flattened",
                "K": "post K projection/head reshape/QK-norm and pre RoPE, all 8 KV heads flattened",
                "V": "post value projection/head reshape, all 8 KV heads flattened",
                "O": "actual attention kernel output before o_proj, all 16 query heads flattened",
                "MLP": "generation MLP sublayer output before block residual addition",
            },
        }
        (self.output_dir / "validation.json").write_text(
            json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return profile
