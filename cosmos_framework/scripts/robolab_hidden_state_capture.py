# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Disk-backed GEN hidden-state capture for RoboLab action-policy experiments."""

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
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image

GEN_HIDDEN_RAW_FILE = "gen_hidden_state_raw.bin"
GEN_HIDDEN_PROFILE_FILE = "gen_hidden_state_profile.json"
CAPTURE_METADATA_FILE = "metadata.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_task_key(prompt: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:64]
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    return f"{slug or 'task'}_{digest}"


@dataclass(frozen=True)
class CaptureSelection:
    prompt: str
    task_key: str
    chunk_index: int
    seed: int
    partial_dir: Path
    final_dir: Path


class HiddenStateCapturePlanner:
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
            raise ValueError(
                "Hidden-state capture output directory must be new or empty: "
                f"{self.output_root}"
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 1,
            "created_utc": _utc_now(),
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

    def select(self, *, prompt: str, seed: int) -> CaptureSelection | None:
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
            raise FileExistsError(
                "Hidden-state capture artifact already exists; use a fresh experiment directory: "
                f"{final_dir}"
            )
        task_dir.mkdir(parents=True, exist_ok=True)
        partial_dir.mkdir()
        return CaptureSelection(
            prompt=prompt,
            task_key=task_key,
            chunk_index=chunk_index,
            seed=int(seed),
            partial_dir=partial_dir,
            final_dir=final_dir,
        )

    def complete(self, selection: CaptureSelection, metadata: Mapping[str, Any]) -> None:
        selection.partial_dir.rename(selection.final_dir)
        manifest_entry = {
            "task_key": selection.task_key,
            "prompt": selection.prompt,
            "chunk_index": selection.chunk_index,
            "seed": selection.seed,
            "artifact_dir": str(selection.final_dir),
            "raw_shape": list(metadata["gen_hidden_state_profile"]["raw_shape"]),
            "completed_utc": _utc_now(),
        }
        with (self.output_root / "manifest.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")

    def mark_failed(self, selection: CaptureSelection, exc: BaseException) -> None:
        if selection.partial_dir.is_dir():
            (selection.partial_dir / "FAILED.txt").write_text(
                f"{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )


class GenHiddenStateCollector:
    """Stream complete GEN vision-token block outputs into a CPU mmap tensor."""

    def __init__(
        self,
        *,
        torch: Any,
        net: Any,
        guidance: float,
        num_steps: int,
        output_dir: Path,
        disk_reserve_bytes: int,
    ) -> None:
        self.torch = torch
        self.net = net
        self.guidance = float(guidance)
        self.num_steps = int(num_steps)
        self.expected_num_calls = self.num_steps if self.guidance == 1.0 else 2 * self.num_steps
        self.output_dir = Path(output_dir)
        self.raw_path = self.output_dir / GEN_HIDDEN_RAW_FILE
        self.disk_reserve_bytes = int(disk_reserve_bytes)

        language_model = getattr(net, "language_model", None)
        decoder = getattr(language_model, "model", None)
        layers = getattr(decoder, "layers", None)
        if layers is None:
            raise RuntimeError("Could not locate net.language_model.model.layers for hidden-state capture")
        self.layers = list(layers)
        if not self.layers:
            raise RuntimeError("GEN hidden-state capture found no transformer blocks")
        self.temporal_compression_factor = int(
            getattr(getattr(net, "config", None), "temporal_compression_factor_vision", 4)
        )

        self._handles: list[Any] = []
        self._current: dict[str, Any] | None = None
        self._forward_call_index = 0
        self._raw_tensor: Any | None = None
        self._raw_dtype: Any | None = None
        self._raw_shape: tuple[int, ...] | None = None
        self._raw_num_bytes = 0
        self._latent_shape_thw: tuple[int, int, int] | None = None
        self._hidden_size: int | None = None
        self._written_positions: set[tuple[int, int]] = set()
        self._timesteps_by_call: dict[int, float] = {}

    def __enter__(self) -> "GenHiddenStateCollector":
        self._handles.append(self.net.register_forward_pre_hook(self._network_pre_hook, with_kwargs=True))
        self._handles.append(self.net.register_forward_hook(self._network_post_hook, with_kwargs=True))
        for layer_index, layer in enumerate(self.layers):
            self._handles.append(
                layer.register_forward_hook(self._make_layer_hook(layer_index), with_kwargs=True)
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc, traceback
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._current = None
        if exc_type is not None:
            self.cleanup()

    def cleanup(self) -> None:
        self._raw_tensor = None
        if self.raw_path.exists():
            self.raw_path.unlink()

    def _allocate_raw_tensor(self, *, dtype: Any, hidden_size: int) -> None:
        if self._raw_tensor is not None:
            return
        if dtype not in (self.torch.bfloat16, self.torch.float16):
            raise RuntimeError(
                "GEN hidden-state capture expects bfloat16 or float16 block outputs, "
                f"got {dtype}"
            )
        if self._latent_shape_thw is None:
            raise RuntimeError("Vision token geometry was not initialized before layer capture")
        latent_frames, patch_height, patch_width = self._latent_shape_thw
        self._hidden_size = int(hidden_size)
        self._raw_shape = (
            self.expected_num_calls,
            len(self.layers),
            latent_frames,
            patch_height * patch_width,
            self._hidden_size,
        )
        numel = math.prod(self._raw_shape)
        self._raw_num_bytes = numel * self.torch.empty((), dtype=dtype).element_size()
        free_bytes = shutil.disk_usage(self.output_dir).free
        required_bytes = self._raw_num_bytes + self.disk_reserve_bytes
        if free_bytes < required_bytes:
            raise RuntimeError(
                "Insufficient disk space for GEN hidden-state capture: "
                f"raw={self._raw_num_bytes / (1 << 30):.2f} GiB, "
                f"reserve={self.disk_reserve_bytes / (1 << 30):.2f} GiB, "
                f"free={free_bytes / (1 << 30):.2f} GiB, output={self.output_dir}"
            )
        with self.raw_path.open("w+b") as handle:
            handle.truncate(self._raw_num_bytes)
        self._raw_tensor = self.torch.from_file(
            str(self.raw_path),
            shared=True,
            size=numel,
            dtype=dtype,
        ).reshape(self._raw_shape)
        self._raw_dtype = dtype

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
        vision = getattr(packed_sequence, "vision", None)
        token_shapes = getattr(vision, "token_shapes", None)
        sequence_indexes = getattr(vision, "sequence_indexes", None)
        if vision is None or not token_shapes or not self.torch.is_tensor(sequence_indexes):
            raise RuntimeError("GEN hidden-state capture requires packed vision tokens")
        if len(token_shapes) != 1:
            raise RuntimeError(
                "GEN hidden-state capture requires one vision item per request, "
                f"got {len(token_shapes)}"
            )

        latent_shape_thw = tuple(map(int, token_shapes[0]))
        if len(latent_shape_thw) != 3:
            raise RuntimeError(f"Vision token shape must be [T,H,W], got {latent_shape_thw}")
        if self._latent_shape_thw is None:
            self._latent_shape_thw = latent_shape_thw
        elif latent_shape_thw != self._latent_shape_thw:
            raise RuntimeError(
                "Vision token shape changed during hidden-state capture: "
                f"expected={self._latent_shape_thw}, got={latent_shape_thw}"
            )
        latent_frames, patch_height, patch_width = latent_shape_thw
        tokens_per_frame = patch_height * patch_width
        vision_indexes = [int(value) for value in sequence_indexes.detach().cpu().tolist()]
        expected_tokens = latent_frames * tokens_per_frame
        if len(vision_indexes) != expected_tokens:
            raise RuntimeError(
                "Vision token geometry mismatch: "
                f"indexes={len(vision_indexes)}, expected={expected_tokens}, shape={latent_shape_thw}"
            )

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
        try:
            vision_positions = tuple(full_position[index] for index in vision_indexes)
        except KeyError as exc:
            raise RuntimeError(
                f"Vision token {int(exc.args[0])} is not in the GEN/full-attention sequence"
            ) from exc

        call_index = self._forward_call_index
        self._forward_call_index += 1
        if call_index >= self.expected_num_calls:
            raise RuntimeError(
                "GEN hidden-state capture received too many forward calls: "
                f"expected={self.expected_num_calls}, current={call_index}"
            )
        timesteps = getattr(vision, "timesteps", None)
        timestep = None
        if self.torch.is_tensor(timesteps) and timesteps.numel() > 0:
            timestep = timesteps.reshape(-1)[0].detach()
        self._timesteps_by_call[call_index] = (
            float("nan") if timestep is None else float(timestep.float().cpu())
        )
        self._current = {
            "forward_call_index": call_index,
            "vision_positions": vision_positions,
            "latent_frames": latent_frames,
            "tokens_per_frame": tokens_per_frame,
            "positions_by_device": {},
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

    def _make_layer_hook(self, layer_index: int) -> Callable[..., None]:
        def capture(
            module: Any,
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
            output: Any,
        ) -> None:
            del module, args, kwargs
            current = self._current
            if current is None:
                return
            output_pack = output[0] if isinstance(output, tuple) else output
            if not isinstance(output_pack, Mapping) or "full_only_seq" not in output_pack:
                raise RuntimeError(f"Transformer block {layer_index} did not return a GEN SequencePack")
            gen_hidden = output_pack["full_only_seq"]
            if not self.torch.is_tensor(gen_hidden) or gen_hidden.ndim != 2:
                raise RuntimeError(
                    f"Transformer block {layer_index} GEN hidden must be [tokens,hidden], "
                    f"got {getattr(gen_hidden, 'shape', None)}"
                )
            actual_tokens = int(output_pack.get("_num_full_tokens", gen_hidden.shape[0]))
            vision_positions = current["vision_positions"]
            if not vision_positions or max(vision_positions) >= actual_tokens:
                raise RuntimeError(
                    f"Transformer block {layer_index} vision positions exceed GEN token count {actual_tokens}"
                )
            device_key = str(gen_hidden.device)
            positions = current["positions_by_device"].get(device_key)
            if positions is None:
                positions = self.torch.tensor(
                    vision_positions,
                    dtype=self.torch.long,
                    device=gen_hidden.device,
                )
                current["positions_by_device"][device_key] = positions
            frame_hidden = gen_hidden.index_select(0, positions).reshape(
                current["latent_frames"],
                current["tokens_per_frame"],
                gen_hidden.shape[-1],
            )
            self._allocate_raw_tensor(dtype=frame_hidden.dtype, hidden_size=int(frame_hidden.shape[-1]))
            if frame_hidden.dtype != self._raw_dtype or int(frame_hidden.shape[-1]) != self._hidden_size:
                raise RuntimeError("GEN hidden-state dtype or hidden size changed during capture")
            position = (int(current["forward_call_index"]), int(layer_index))
            if position in self._written_positions:
                raise RuntimeError(
                    "GEN hidden-state capture saw the same call/block twice: "
                    f"call={position[0]}, block={position[1]}"
                )
            self._raw_tensor[position[0], position[1]].copy_(frame_hidden.detach(), non_blocking=False)
            self._written_positions.add(position)

        return capture

    def finish(self) -> dict[str, Any]:
        try:
            if (
                self._raw_tensor is None
                or self._raw_shape is None
                or self._raw_dtype is None
                or self._latent_shape_thw is None
                or self._hidden_size is None
            ):
                raise RuntimeError("GEN hidden-state collector captured no transformer outputs")
            expected_positions = self.expected_num_calls * len(self.layers)
            if (
                self._forward_call_index != self.expected_num_calls
                or len(self._written_positions) != expected_positions
            ):
                raise RuntimeError(
                    "Incomplete GEN hidden-state capture: "
                    f"calls={self._forward_call_index}/{self.expected_num_calls}, "
                    f"call-blocks={len(self._written_positions)}/{expected_positions}"
                )
            profile = {
                "raw_file": GEN_HIDDEN_RAW_FILE,
                "raw_shape": list(self._raw_shape),
                "raw_dtype": str(self._raw_dtype),
                "raw_num_bytes": self._raw_num_bytes,
                "axis_names": [
                    "forward_call",
                    "transformer_block",
                    "latent_frame",
                    "spatial_token",
                    "hidden_channel",
                ],
                "num_forward_calls": self.expected_num_calls,
                "num_layers": len(self.layers),
                "latent_shape_thw": list(self._latent_shape_thw),
                "hidden_size": self._hidden_size,
                "temporal_compression_factor": self.temporal_compression_factor,
                "cfg_branch_by_call": [
                    "conditional" if self.guidance == 1.0 or index % 2 == 0 else "unconditional"
                    for index in range(self.expected_num_calls)
                ],
                "sampler_step_by_call": [
                    index if self.guidance == 1.0 else index // 2
                    for index in range(self.expected_num_calls)
                ],
                "timestep_by_call": [
                    self._timesteps_by_call[index] for index in range(self.expected_num_calls)
                ],
            }
            self._raw_tensor = None
            with self.raw_path.open("rb") as handle:
                os.fsync(handle.fileno())
            return profile
        except Exception:
            self.cleanup()
            raise


def _to_uint8_frames(torch: Any, pred_video: Any) -> np.ndarray:
    video = pred_video.detach().cpu().float()
    if video.ndim == 5 and video.shape[0] == 1:
        video = video.squeeze(0)
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError(f"Decoded vision must have shape [C,T,H,W], got {tuple(video.shape)}")
    if float(video.min()) < 0.0:
        video = (video + 1.0) / 2.0
    return (
        (video.clamp(0.0, 1.0) * 255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 3, 0)
        .contiguous()
        .numpy()
    )


def save_capture_artifact(
    *,
    torch: Any,
    selection: CaptureSelection,
    profile: Mapping[str, Any],
    vision_latent: Any,
    pred_video: Any,
    conditioning_image: np.ndarray,
    fps: float,
    action_chunk_size: int,
) -> dict[str, Any]:
    """Save one selected action chunk in the format consumed by the analysis scripts."""

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

    latent_frames, patch_height, patch_width = map(int, profile["latent_shape_thw"])
    temporal_compression_factor = int(profile["temporal_compression_factor"])
    decoded_frame_ranges = [[0, 0]]
    for latent_index in range(1, latent_frames):
        decoded_frame_ranges.append(
            [
                (latent_index - 1) * temporal_compression_factor + 1,
                min(latent_index * temporal_compression_factor, len(frames) - 1),
            ]
        )
    profile_summary = {
        "schema_version": 2,
        "scope": "Raw GEN vision-token outputs after every transformer block",
        **dict(profile),
        "decoded_frame_ranges_by_latent_frame": decoded_frame_ranges,
        "storage_order": "C contiguous",
        "spatial_token_rule": (
            "spatial_token = patch_y * patch_width + patch_x; "
            f"patch_height={patch_height}, patch_width={patch_width}"
        ),
        "capture_rule": (
            "Each [forward_call, transformer_block] slice stores all vision-token hidden states; "
            "sampler_step_by_call and cfg_branch_by_call map calls to denoising steps and CFG branches."
        ),
        "files": {"raw": GEN_HIDDEN_RAW_FILE, "summary": GEN_HIDDEN_PROFILE_FILE},
    }
    (output_dir / GEN_HIDDEN_PROFILE_FILE).write_text(
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
        "gen_hidden_state_profile": profile_summary,
        "completed_utc": _utc_now(),
    }
    (output_dir / CAPTURE_METADATA_FILE).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata
