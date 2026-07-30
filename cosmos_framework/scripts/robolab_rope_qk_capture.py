# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Selective pre/post-RoPE GEN Q/K capture for RoboLab experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from cosmos_framework.scripts.robolab_hidden_state_capture import _to_uint8_frames

ROPE_QK_PROFILE_FILE = "rope_qk_profile.json"
ROPE_QK_METADATA_FILE = "metadata.json"
ROPE_QK_POSITION_FILE = "position_ids.pt"
ROPE_QK_COS_SIN_FILE = "rope_cos_sin.pt"
ROPE_QK_RAW_FILES = {
    "q_raw": "q_raw.bin",
    "k_raw": "k_raw.bin",
    "q_rope": "q_rope.bin",
    "k_rope": "k_rope.bin",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_task_key(prompt: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:64]
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    return f"{slug or 'task'}_{digest}"


@dataclass(frozen=True)
class RopeQKCaptureSelection:
    prompt: str
    task_key: str
    chunk_index: int
    seed: int
    partial_dir: Path
    final_dir: Path


class RopeQKCapturePlanner:
    """Select configured request/chunk indexes independently for every prompt."""

    def __init__(
        self,
        *,
        output_root: Path,
        chunk_indices: list[int],
        experiment_metadata: Mapping[str, Any],
    ) -> None:
        self.output_root = Path(output_root).expanduser().absolute()
        self.chunk_indices = tuple(sorted(int(value) for value in chunk_indices))
        self._chunk_index_set = set(self.chunk_indices)
        self._request_count_by_prompt: dict[str, int] = {}
        self._lock = threading.Lock()

        if self.output_root.exists() and any(self.output_root.iterdir()):
            raise ValueError(f"RoPE Q/K capture output directory must be new or empty: {self.output_root}")
        self.output_root.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 1,
            "created_utc": _utc_now(),
            "capture_type": "gen_rope_qk",
            "chunk_index_rule": (
                "Zero-based WebSocket policy request index, counted independently per exact prompt; "
                "one request produces one action chunk."
            ),
            "selected_chunk_indices": list(self.chunk_indices),
            **dict(experiment_metadata),
        }
        (self.output_root / "experiment.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def select(self, *, prompt: str, seed: int) -> RopeQKCaptureSelection | None:
        with self._lock:
            chunk_index = self._request_count_by_prompt.get(prompt, 0)
            self._request_count_by_prompt[prompt] = chunk_index + 1
        if chunk_index not in self._chunk_index_set:
            return None

        task_key = _safe_task_key(prompt)
        task_dir = self.output_root / f"task_{task_key}"
        final_dir = task_dir / f"chunk_{chunk_index:06d}"
        partial_dir = task_dir / f"chunk_{chunk_index:06d}.partial"
        if final_dir.exists() or partial_dir.exists():
            raise FileExistsError(f"RoPE Q/K artifact already exists; use a fresh experiment directory: {final_dir}")
        task_dir.mkdir(parents=True, exist_ok=True)
        partial_dir.mkdir()
        return RopeQKCaptureSelection(
            prompt=prompt,
            task_key=task_key,
            chunk_index=chunk_index,
            seed=int(seed),
            partial_dir=partial_dir,
            final_dir=final_dir,
        )

    def complete(self, selection: RopeQKCaptureSelection, metadata: Mapping[str, Any]) -> None:
        selection.partial_dir.rename(selection.final_dir)
        profile = metadata["rope_qk_profile"]
        manifest_entry = {
            "task_key": selection.task_key,
            "prompt": selection.prompt,
            "chunk_index": selection.chunk_index,
            "seed": selection.seed,
            "artifact_dir": str(selection.final_dir),
            "selected_steps": profile["selected_steps"],
            "selected_blocks": profile["selected_blocks"],
            "selected_branches": profile["selected_branches"],
            "completed_utc": _utc_now(),
        }
        with (self.output_root / "manifest.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")

    def mark_failed(self, selection: RopeQKCaptureSelection, exc: BaseException) -> None:
        if selection.partial_dir.is_dir():
            (selection.partial_dir / "FAILED.txt").write_text(
                f"{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )


def _contiguous_ranges(indexes: list[int]) -> list[list[int]]:
    if not indexes:
        return []
    ranges: list[list[int]] = []
    start = previous = indexes[0]
    for index in indexes[1:]:
        if index != previous + 1:
            ranges.append([start, previous + 1])
            start = index
        previous = index
    ranges.append([start, previous + 1])
    return ranges


class GenRopeQKCollector:
    """Capture normalized GEN Q/K immediately before and after RoPE."""

    def __init__(
        self,
        *,
        torch: Any,
        net: Any,
        guidance: float,
        num_steps: int,
        selected_steps: list[int],
        selected_blocks: list[int],
        selected_branches: list[str],
        output_dir: Path,
        disk_reserve_bytes: int,
    ) -> None:
        self.torch = torch
        self.net = net
        self.guidance = float(guidance)
        self.num_steps = int(num_steps)
        self.selected_steps = tuple(sorted(int(value) for value in selected_steps))
        self.selected_blocks = tuple(sorted(int(value) for value in selected_blocks))
        self.selected_branches = tuple(selected_branches)
        self._step_slot = {value: index for index, value in enumerate(self.selected_steps)}
        self._block_slot = {value: index for index, value in enumerate(self.selected_blocks)}
        self.output_dir = Path(output_dir)
        self.disk_reserve_bytes = int(disk_reserve_bytes)

        language_model = getattr(net, "language_model", None)
        decoder = getattr(language_model, "model", None)
        layers = getattr(decoder, "layers", None)
        if layers is None:
            raise RuntimeError("Could not locate net.language_model.model.layers for RoPE Q/K capture")
        self.layers = list(layers)
        if not self.layers:
            raise RuntimeError("RoPE Q/K capture found no transformer blocks")
        invalid_blocks = [index for index in self.selected_blocks if index >= len(self.layers)]
        if invalid_blocks:
            raise ValueError(f"Selected blocks exceed model layer count {len(self.layers)}: {invalid_blocks}")

        self._handles: list[Any] = []
        self._attention_modules: list[Any] = []
        self._current: dict[str, Any] | None = None
        self._forward_call_index = 0
        self._raw_tensors: dict[str, Any] = {}
        self._raw_shapes: dict[str, tuple[int, ...]] = {}
        self._raw_dtype: Any | None = None
        self._raw_num_bytes = 0
        self._num_gen_tokens: int | None = None
        self._num_q_heads: int | None = None
        self._num_kv_heads: int | None = None
        self._head_dim: int | None = None
        self._written_positions: set[tuple[int, int]] = set()
        self._selected_calls: dict[int, dict[str, Any]] = {}
        self._position_ids_by_step: dict[int, Any] = {}
        self._cos_by_step: dict[int, Any] = {}
        self._sin_by_step: dict[int, Any] = {}
        self._token_layout: dict[str, Any] | None = None

    def __enter__(self) -> "GenRopeQKCollector":
        self._handles.append(self.net.register_forward_pre_hook(self._network_pre_hook, with_kwargs=True))
        self._handles.append(self.net.register_forward_hook(self._network_post_hook, with_kwargs=True))
        for block_index in self.selected_blocks:
            attention = getattr(self.layers[block_index], "self_attn", None)
            if attention is None or not hasattr(attention, "_rope_qk_capture_callback"):
                raise RuntimeError(f"Transformer block {block_index} does not expose the RoPE Q/K capture hook")
            if attention._rope_qk_capture_callback is not None:
                raise RuntimeError(f"Transformer block {block_index} already has a RoPE Q/K capture callback")
            attention._rope_qk_capture_callback = self._capture_qk
            self._attention_modules.append(attention)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc, traceback
        for attention in self._attention_modules:
            attention._rope_qk_capture_callback = None
        self._attention_modules.clear()
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._current = None
        if exc_type is not None:
            self.cleanup()

    def cleanup(self) -> None:
        self._raw_tensors.clear()
        for file_name in (*ROPE_QK_RAW_FILES.values(), ROPE_QK_POSITION_FILE, ROPE_QK_COS_SIN_FILE):
            path = self.output_dir / file_name
            if path.exists():
                path.unlink()

    def _call_semantics(self, call_index: int) -> tuple[int, str]:
        if self.guidance == 1.0:
            return call_index, "conditional"
        return call_index // 2, "conditional" if call_index % 2 == 0 else "unconditional"

    def _network_pre_hook(
        self,
        module: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> None:
        del module
        und_only = bool(kwargs.get("und_only", args[2] if len(args) > 2 else False))
        if und_only:
            self._current = None
            return

        packed_sequence = args[0] if args else kwargs.get("packed_seq")
        call_index = self._forward_call_index
        self._forward_call_index += 1
        sampler_step, branch = self._call_semantics(call_index)
        if sampler_step not in self._step_slot or branch not in self.selected_branches:
            self._current = None
            return

        full_indexes: list[int] = []
        offset = 0
        for mode, split_len in zip(
            packed_sequence.attn_modes,
            packed_sequence.split_lens,
            strict=True,
        ):
            split_len = int(split_len)
            if mode == "full":
                full_indexes.extend(range(offset, offset + split_len))
            offset += split_len
        full_position = {original: position for position, original in enumerate(full_indexes)}

        vision = getattr(packed_sequence, "vision", None)
        action = getattr(packed_sequence, "action", None)
        if vision is None or not self.torch.is_tensor(getattr(vision, "sequence_indexes", None)):
            raise RuntimeError("RoPE Q/K capture requires packed vision tokens")
        vision_original = [int(value) for value in vision.sequence_indexes.detach().cpu().tolist()]
        action_indexes = getattr(action, "sequence_indexes", None)
        action_original = (
            [int(value) for value in action_indexes.detach().cpu().tolist()]
            if self.torch.is_tensor(action_indexes)
            else []
        )
        try:
            vision_positions = [full_position[index] for index in vision_original]
            action_positions = [full_position[index] for index in action_original]
        except KeyError as exc:
            raise RuntimeError(f"Modality token {int(exc.args[0])} is not in the GEN sequence") from exc

        position_ids = packed_sequence.position_ids
        if not self.torch.is_tensor(position_ids):
            raise RuntimeError("RoPE Q/K capture requires finalized position_ids")
        full_indexes_tensor = self.torch.tensor(full_indexes, dtype=self.torch.long, device=position_ids.device)
        if position_ids.ndim == 1:
            gen_position_ids = position_ids.index_select(0, full_indexes_tensor).unsqueeze(0)
        elif position_ids.ndim == 2:
            gen_position_ids = position_ids.index_select(1, full_indexes_tensor)
        else:
            raise RuntimeError(f"Expected position_ids [N] or [axes,N], got {tuple(position_ids.shape)}")
        gen_position_ids = gen_position_ids.detach().cpu().clone()

        token_shapes = getattr(vision, "token_shapes", None)
        if not token_shapes or len(token_shapes) != 1:
            raise RuntimeError(f"Expected one vision token shape, got {token_shapes}")
        latent_shape_thw = [int(value) for value in token_shapes[0]]
        if math.prod(latent_shape_thw) != len(vision_positions):
            raise RuntimeError(
                f"Vision token shape {latent_shape_thw} does not match {len(vision_positions)} positions"
            )
        action_shapes = [[int(value) for value in shape] for shape in (getattr(action, "token_shapes", None) or [])]
        token_layout = {
            "num_gen_tokens": len(full_indexes),
            "vision": {
                "original_sequence_indexes": vision_original,
                "gen_positions": vision_positions,
                "gen_position_ranges": _contiguous_ranges(vision_positions),
                "latent_shape_thw": latent_shape_thw,
                "spatial_token_rule": (
                    "vision token order is latent_t major, then patch_y, then patch_x; "
                    f"patch_height={latent_shape_thw[1]}, patch_width={latent_shape_thw[2]}"
                ),
            },
            "action": {
                "original_sequence_indexes": action_original,
                "gen_positions": action_positions,
                "gen_position_ranges": _contiguous_ranges(action_positions),
                "token_shapes": action_shapes,
            },
        }
        if self._token_layout is None:
            self._token_layout = token_layout
        elif token_layout != self._token_layout:
            raise RuntimeError("GEN token layout changed across selected denoising calls")

        timesteps = getattr(vision, "timesteps", None)
        timestep = float("nan")
        if self.torch.is_tensor(timesteps) and timesteps.numel() > 0:
            timestep = float(timesteps.reshape(-1)[0].detach().float().cpu())
        self._selected_calls[sampler_step] = {
            "forward_call_index": call_index,
            "sampler_step": sampler_step,
            "branch": branch,
            "timestep": timestep,
        }
        self._position_ids_by_step[sampler_step] = gen_position_ids
        self._current = {
            "forward_call_index": call_index,
            "sampler_step": sampler_step,
            "branch": branch,
            "num_gen_tokens": len(full_indexes),
        }

    def _network_post_hook(
        self,
        module: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: Any,
    ) -> None:
        del module, args, kwargs, output
        self._current = None

    def _allocate(
        self,
        *,
        dtype: Any,
        num_gen_tokens: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        if self._raw_tensors:
            return
        if dtype not in (self.torch.bfloat16, self.torch.float16):
            raise RuntimeError(f"RoPE Q/K capture expects bfloat16 or float16, got {dtype}")
        self._raw_dtype = dtype
        self._num_gen_tokens = num_gen_tokens
        self._num_q_heads = num_q_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        prefix = (len(self.selected_steps), len(self.selected_blocks), num_gen_tokens)
        self._raw_shapes = {
            "q_raw": (*prefix, num_q_heads, head_dim),
            "q_rope": (*prefix, num_q_heads, head_dim),
            "k_raw": (*prefix, num_kv_heads, head_dim),
            "k_rope": (*prefix, num_kv_heads, head_dim),
        }
        element_size = self.torch.empty((), dtype=dtype).element_size()
        self._raw_num_bytes = sum(math.prod(shape) * element_size for shape in self._raw_shapes.values())
        free_bytes = shutil.disk_usage(self.output_dir).free
        required_bytes = self._raw_num_bytes + self.disk_reserve_bytes
        if free_bytes < required_bytes:
            raise RuntimeError(
                "Insufficient disk space for RoPE Q/K capture: "
                f"raw={self._raw_num_bytes / (1 << 30):.2f} GiB, "
                f"reserve={self.disk_reserve_bytes / (1 << 30):.2f} GiB, "
                f"free={free_bytes / (1 << 30):.2f} GiB"
            )
        for name, shape in self._raw_shapes.items():
            path = self.output_dir / ROPE_QK_RAW_FILES[name]
            numel = math.prod(shape)
            with path.open("w+b") as handle:
                handle.truncate(numel * element_size)
            self._raw_tensors[name] = self.torch.from_file(
                str(path),
                shared=True,
                size=numel,
                dtype=dtype,
            ).reshape(shape)

    def _capture_qk(
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
        current = self._current
        if current is None or layer_index not in self._block_slot:
            return
        num_gen_tokens = int(current["num_gen_tokens"])
        tensors = {"q_raw": q_raw, "k_raw": k_raw, "q_rope": q_rope, "k_rope": k_rope}
        for name, tensor in tensors.items():
            if not self.torch.is_tensor(tensor) or tensor.ndim != 3:
                raise RuntimeError(f"{name} must be [tokens,heads,head_dim], got {getattr(tensor, 'shape', None)}")
            if tensor.shape[0] < num_gen_tokens:
                raise RuntimeError(f"{name} has {tensor.shape[0]} tokens, expected at least {num_gen_tokens}")
        self._allocate(
            dtype=q_raw.dtype,
            num_gen_tokens=num_gen_tokens,
            num_q_heads=int(q_raw.shape[1]),
            num_kv_heads=int(k_raw.shape[1]),
            head_dim=int(q_raw.shape[2]),
        )
        expected = (
            self._num_gen_tokens,
            self._num_q_heads,
            self._num_kv_heads,
            self._head_dim,
            self._raw_dtype,
        )
        actual = (
            num_gen_tokens,
            int(q_raw.shape[1]),
            int(k_raw.shape[1]),
            int(q_raw.shape[2]),
            q_raw.dtype,
        )
        if actual != expected:
            raise RuntimeError(f"RoPE Q/K tensor geometry changed: expected={expected}, got={actual}")

        step = int(current["sampler_step"])
        step_slot = self._step_slot[step]
        block_slot = self._block_slot[layer_index]
        position = (step, layer_index)
        if position in self._written_positions:
            raise RuntimeError(f"RoPE Q/K saw step/block twice: step={step}, block={layer_index}")
        for name, tensor in tensors.items():
            cpu = tensor[:num_gen_tokens].detach().to(device="cpu", copy=True)
            self._raw_tensors[name][step_slot, block_slot].copy_(cpu)
        if step not in self._cos_by_step:
            self._cos_by_step[step] = cos[:num_gen_tokens].detach().to(device="cpu", copy=True)
            self._sin_by_step[step] = sin[:num_gen_tokens].detach().to(device="cpu", copy=True)
        self._written_positions.add(position)

    def finish(self) -> dict[str, Any]:
        try:
            expected_positions = len(self.selected_steps) * len(self.selected_blocks)
            if (
                not self._raw_tensors
                or self._raw_dtype is None
                or self._num_gen_tokens is None
                or self._num_q_heads is None
                or self._num_kv_heads is None
                or self._head_dim is None
                or self._token_layout is None
            ):
                raise RuntimeError("RoPE Q/K collector captured no tensors")
            if len(self._written_positions) != expected_positions:
                raise RuntimeError(
                    f"Incomplete RoPE Q/K capture: step-blocks={len(self._written_positions)}/{expected_positions}"
                )
            if set(self._selected_calls) != set(self.selected_steps):
                raise RuntimeError(
                    f"Missing selected denoising calls: got={sorted(self._selected_calls)}, "
                    f"expected={list(self.selected_steps)}"
                )

            position_payload = {
                "selected_steps": list(self.selected_steps),
                "axis_names": ["selected_step", "mrope_axis", "gen_token"],
                "mrope_axis_names": ["temporal", "height", "width"],
                "position_ids": self.torch.stack([self._position_ids_by_step[step] for step in self.selected_steps]),
            }
            cos_sin_payload = {
                "selected_steps": list(self.selected_steps),
                "axis_names": ["selected_step", "gen_token", "head_dim"],
                "cos": self.torch.stack([self._cos_by_step[step] for step in self.selected_steps]),
                "sin": self.torch.stack([self._sin_by_step[step] for step in self.selected_steps]),
            }
            self.torch.save(position_payload, self.output_dir / ROPE_QK_POSITION_FILE)
            self.torch.save(cos_sin_payload, self.output_dir / ROPE_QK_COS_SIN_FILE)

            profile = {
                "schema_version": 1,
                "scope": "GEN Q/K after projection and QK norm, immediately before and after RoPE",
                "selected_steps": list(self.selected_steps),
                "selected_blocks": list(self.selected_blocks),
                "selected_branches": list(self.selected_branches),
                "selected_calls": [self._selected_calls[step] for step in self.selected_steps],
                "raw_dtype": str(self._raw_dtype),
                "raw_num_bytes": self._raw_num_bytes,
                "num_gen_tokens": self._num_gen_tokens,
                "num_q_heads": self._num_q_heads,
                "num_kv_heads": self._num_kv_heads,
                "q_heads_per_kv_head": self._num_q_heads // self._num_kv_heads,
                "head_dim": self._head_dim,
                "axis_names": ["selected_step", "selected_block", "gen_token", "attention_head", "head_dim"],
                "raw_shapes": {name: list(shape) for name, shape in self._raw_shapes.items()},
                "raw_files": dict(ROPE_QK_RAW_FILES),
                "position_ids_file": ROPE_QK_POSITION_FILE,
                "rope_cos_sin_file": ROPE_QK_COS_SIN_FILE,
                "token_layout": self._token_layout,
                "capture_definition": (
                    "q_raw/k_raw are normalized GEN Q/K passed into apply_rotary_pos_emb; "
                    "q_rope/k_rope are that function's outputs."
                ),
                "storage_order": "C contiguous",
            }
            self._raw_tensors.clear()
            for file_name in ROPE_QK_RAW_FILES.values():
                with (self.output_dir / file_name).open("rb") as handle:
                    os.fsync(handle.fileno())
            return profile
        except Exception:
            self.cleanup()
            raise


def save_rope_qk_capture_artifact(
    *,
    torch: Any,
    selection: RopeQKCaptureSelection,
    profile: Mapping[str, Any],
    vision_latent: Any,
    pred_video: Any,
    conditioning_image: np.ndarray,
    fps: float,
    action_chunk_size: int,
) -> dict[str, Any]:
    """Save selected Q/K tensors together with the matching decoded rollout."""

    output_dir = selection.partial_dir
    latent_cpu = vision_latent.detach().cpu()
    future_latent_cpu = latent_cpu[:, :, 1:].contiguous() if latent_cpu.ndim == 5 else latent_cpu
    torch.save(latent_cpu, output_dir / "denoised_vision_latent.pt")
    torch.save(future_latent_cpu, output_dir / "future_vision_latent.pt")

    frames = _to_uint8_frames(torch, pred_video)
    future_frames_dir = output_dir / "predicted_future_frames"
    future_frames_dir.mkdir()
    for frame_index, frame in enumerate(frames[1:], start=1):
        Image.fromarray(frame).save(future_frames_dir / f"frame_{frame_index:03d}.png")
    conditioning = np.asarray(conditioning_image)
    if conditioning.ndim != 3 or conditioning.shape[-1] != 3:
        raise ValueError(f"conditioning_image must be [H,W,3], got {conditioning.shape}")
    if conditioning.dtype != np.uint8:
        conditioning = np.clip(conditioning, 0, 255).astype(np.uint8)
    Image.fromarray(np.ascontiguousarray(conditioning)).save(output_dir / "conditioning_observation.png")

    profile_summary = dict(profile)
    (output_dir / ROPE_QK_PROFILE_FILE).write_text(
        json.dumps(profile_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": 1,
        "artifact_dir": str(selection.final_dir),
        "task_key": selection.task_key,
        "prompt": selection.prompt,
        "chunk_index": selection.chunk_index,
        "chunk_index_base": 0,
        "sim_action_step_range": [
            selection.chunk_index * int(action_chunk_size),
            (selection.chunk_index + 1) * int(action_chunk_size) - 1,
        ],
        "seed": selection.seed,
        "fps": float(fps),
        "vision_latent_shape": list(latent_cpu.shape),
        "future_vision_latent_shape": list(future_latent_cpu.shape),
        "vision_latent_dtype": str(latent_cpu.dtype),
        "decoded_video_shape": list(pred_video.shape),
        "predicted_frame_count": int(len(frames)),
        "future_frame_count": max(0, int(len(frames)) - 1),
        "future_frame_rule": "Decoded frame 0 is conditioning; predicted_future_frames stores frames 1..T-1.",
        "conditioning_observation_file": "conditioning_observation.png",
        "denoised_latent_file": "denoised_vision_latent.pt",
        "future_latent_file": "future_vision_latent.pt",
        "predicted_future_frames_dir": "predicted_future_frames",
        "rope_qk_profile": profile_summary,
        "completed_utc": _utc_now(),
    }
    (output_dir / ROPE_QK_METADATA_FILE).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata
