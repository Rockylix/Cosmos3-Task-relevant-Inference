# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Disk-backed future-frame block-residual capture for RoboLab policy experiments."""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image

from cosmos_framework.scripts.robolab_hidden_state_capture import (
    CAPTURE_METADATA_FILE,
    CaptureSelection,
    HiddenStateCapturePlanner,
    _to_uint8_frames,
    _utc_now,
)

GEN_BLOCK_RESIDUAL_RAW_FILE = "gen_block_residual_raw.bin"
GEN_BLOCK_RESIDUAL_PROFILE_FILE = "gen_block_residual_profile.json"
RESIDUAL_NORM_FILE = "residual_frobenius_norm.npy"
OUTPUT_NORM_FILE = "block_output_frobenius_norm.npy"
RESIDUAL_OUTPUT_RATIO_FILE = "residual_to_output_ratio.npy"


class BlockResidualCapturePlanner(HiddenStateCapturePlanner):
    """Select chunks and finalize block-residual artifacts."""

    def complete(self, selection: CaptureSelection, metadata: Mapping[str, Any]) -> None:
        selection.partial_dir.rename(selection.final_dir)
        profile = metadata["gen_block_residual_profile"]
        manifest_entry = {
            "task_key": selection.task_key,
            "prompt": selection.prompt,
            "chunk_index": selection.chunk_index,
            "seed": selection.seed,
            "artifact_dir": str(selection.final_dir),
            "raw_shape": list(profile["raw_shape"]),
            "completed_utc": _utc_now(),
        }
        with (self.output_root / "manifest.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")


class GenBlockResidualCollector:
    """Capture ``H_out - H_in`` for future L1..L8 at every GEN block/call."""

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
        self.raw_path = self.output_dir / GEN_BLOCK_RESIDUAL_RAW_FILE
        self.disk_reserve_bytes = int(disk_reserve_bytes)

        language_model = getattr(net, "language_model", None)
        decoder = getattr(language_model, "model", None)
        layers = getattr(decoder, "layers", None)
        if layers is None:
            raise RuntimeError("Could not locate net.language_model.model.layers for residual capture")
        self.layers = list(layers)
        if not self.layers:
            raise RuntimeError("GEN block-residual capture found no transformer blocks")

        self._handles: list[Any] = []
        self._current: dict[str, Any] | None = None
        self._forward_call_index = 0
        self._raw_tensor: Any | None = None
        self._raw_dtype: Any | None = None
        self._raw_shape: tuple[int, ...] | None = None
        self._raw_num_bytes = 0
        self._latent_shape_thw: tuple[int, int, int] | None = None
        self._hidden_size: int | None = None
        self._pending_inputs: dict[tuple[int, int], Any] = {}
        self._written_positions: set[tuple[int, int]] = set()
        self._timesteps_by_call: dict[int, float] = {}
        shape = (self.expected_num_calls, len(self.layers), 0)
        self._residual_norms = np.empty(shape, dtype=np.float64)
        self._output_norms = np.empty(shape, dtype=np.float64)

    def __enter__(self) -> "GenBlockResidualCollector":
        self._handles.append(self.net.register_forward_pre_hook(self._network_pre_hook, with_kwargs=True))
        self._handles.append(self.net.register_forward_hook(self._network_post_hook, with_kwargs=True))
        for layer_index, layer in enumerate(self.layers):
            self._handles.append(
                layer.register_forward_pre_hook(self._make_layer_pre_hook(layer_index), with_kwargs=True)
            )
            self._handles.append(layer.register_forward_hook(self._make_layer_post_hook(layer_index), with_kwargs=True))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc, traceback
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._current = None
        self._pending_inputs.clear()
        if exc_type is not None:
            self.cleanup()

    def cleanup(self) -> None:
        self._raw_tensor = None
        if self.raw_path.exists():
            self.raw_path.unlink()

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
            raise RuntimeError("GEN block-residual capture requires packed vision tokens")
        if len(token_shapes) != 1:
            raise RuntimeError(
                f"GEN block-residual capture requires one vision item per request, got {len(token_shapes)}"
            )

        latent_shape_thw = tuple(map(int, token_shapes[0]))
        if len(latent_shape_thw) != 3:
            raise RuntimeError(f"Vision token shape must be [T,H,W], got {latent_shape_thw}")
        latent_frames, patch_height, patch_width = latent_shape_thw
        if latent_frames < 2:
            raise RuntimeError("Block-residual capture requires condition L0 plus future latent frames")
        if self._latent_shape_thw is None:
            self._latent_shape_thw = latent_shape_thw
            future_frames = latent_frames - 1
            shape = (self.expected_num_calls, len(self.layers), future_frames)
            self._residual_norms = np.full(shape, np.nan, dtype=np.float64)
            self._output_norms = np.full(shape, np.nan, dtype=np.float64)
        elif latent_shape_thw != self._latent_shape_thw:
            raise RuntimeError(
                "Vision token shape changed during block-residual capture: "
                f"expected={self._latent_shape_thw}, got={latent_shape_thw}"
            )

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
            raise RuntimeError(f"Vision token {int(exc.args[0])} is not in the GEN/full-attention sequence") from exc
        # L0 is the first latent frame. It is deliberately excluded from this experiment.
        future_positions = vision_positions[tokens_per_frame:]

        call_index = self._forward_call_index
        self._forward_call_index += 1
        if call_index >= self.expected_num_calls:
            raise RuntimeError(
                "GEN block-residual capture received too many forward calls: "
                f"expected={self.expected_num_calls}, current={call_index}"
            )
        timesteps = getattr(vision, "timesteps", None)
        timestep = None
        if self.torch.is_tensor(timesteps) and timesteps.numel() > 0:
            timestep = timesteps.reshape(-1)[0].detach()
        self._timesteps_by_call[call_index] = float("nan") if timestep is None else float(timestep.float().cpu())
        self._current = {
            "forward_call_index": call_index,
            "future_positions": future_positions,
            "future_frames": latent_frames - 1,
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
        if self._pending_inputs:
            raise RuntimeError(
                f"GEN block-residual capture has unmatched block inputs at network exit: {sorted(self._pending_inputs)}"
            )
        self._current = None

    def _select_future(self, pack: Any, *, layer_index: int) -> Any:
        if not isinstance(pack, Mapping) or "full_only_seq" not in pack:
            raise RuntimeError(f"Transformer block {layer_index} did not expose a GEN SequencePack")
        hidden = pack["full_only_seq"]
        if not self.torch.is_tensor(hidden) or hidden.ndim != 2:
            raise RuntimeError(
                f"Transformer block {layer_index} GEN hidden must be [tokens,hidden], "
                f"got {getattr(hidden, 'shape', None)}"
            )
        current = self._current
        assert current is not None
        actual_tokens = int(pack.get("_num_full_tokens", hidden.shape[0]))
        future_positions = current["future_positions"]
        if not future_positions or max(future_positions) >= actual_tokens:
            raise RuntimeError(
                f"Transformer block {layer_index} future positions exceed GEN token count {actual_tokens}"
            )
        device_key = str(hidden.device)
        positions = current["positions_by_device"].get(device_key)
        if positions is None:
            positions = self.torch.tensor(
                future_positions,
                dtype=self.torch.long,
                device=hidden.device,
            )
            current["positions_by_device"][device_key] = positions
        return hidden.index_select(0, positions).reshape(
            current["future_frames"],
            current["tokens_per_frame"],
            hidden.shape[-1],
        )

    def _make_layer_pre_hook(self, layer_index: int) -> Callable[..., None]:
        def capture(
            module: Any,
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> None:
            del module
            current = self._current
            if current is None:
                return
            input_pack = args[0] if args else kwargs.get("input")
            future_input = self._select_future(input_pack, layer_index=layer_index)
            position = (int(current["forward_call_index"]), int(layer_index))
            if position in self._pending_inputs or position in self._written_positions:
                raise RuntimeError(
                    "GEN block-residual capture saw the same call/block input twice: "
                    f"call={position[0]}, block={position[1]}"
                )
            # Clone before the block so an in-place implementation cannot corrupt H_in.
            self._pending_inputs[position] = future_input.detach().clone()

        return capture

    def _allocate_raw_tensor(self, *, dtype: Any, hidden_size: int) -> None:
        if self._raw_tensor is not None:
            return
        if dtype not in (self.torch.bfloat16, self.torch.float16):
            raise RuntimeError(f"GEN block-residual capture expects bfloat16 or float16 block outputs, got {dtype}")
        if self._latent_shape_thw is None:
            raise RuntimeError("Vision token geometry was not initialized before residual capture")
        latent_frames, patch_height, patch_width = self._latent_shape_thw
        self._hidden_size = int(hidden_size)
        self._raw_shape = (
            self.expected_num_calls,
            len(self.layers),
            latent_frames - 1,
            patch_height * patch_width,
            self._hidden_size,
        )
        numel = math.prod(self._raw_shape)
        self._raw_num_bytes = numel * self.torch.empty((), dtype=dtype).element_size()
        free_bytes = shutil.disk_usage(self.output_dir).free
        required_bytes = self._raw_num_bytes + self.disk_reserve_bytes
        if free_bytes < required_bytes:
            raise RuntimeError(
                "Insufficient disk space for GEN block-residual capture: "
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

    def _make_layer_post_hook(self, layer_index: int) -> Callable[..., None]:
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
            future_output = self._select_future(output_pack, layer_index=layer_index)
            position = (int(current["forward_call_index"]), int(layer_index))
            future_input = self._pending_inputs.pop(position, None)
            if future_input is None:
                raise RuntimeError(
                    "GEN block-residual capture saw output without matching input: "
                    f"call={position[0]}, block={position[1]}"
                )
            if future_input.shape != future_output.shape:
                raise RuntimeError(
                    f"Transformer block {layer_index} changed future hidden shape: "
                    f"input={tuple(future_input.shape)}, output={tuple(future_output.shape)}"
                )
            residual = future_output - future_input
            self._allocate_raw_tensor(dtype=residual.dtype, hidden_size=int(residual.shape[-1]))
            if residual.dtype != self._raw_dtype or int(residual.shape[-1]) != self._hidden_size:
                raise RuntimeError("GEN block-residual dtype or hidden size changed during capture")
            self._raw_tensor[position[0], position[1]].copy_(residual.detach(), non_blocking=False)
            self._residual_norms[position[0], position[1]] = (
                residual.detach().float().square().sum(dim=(-2, -1)).sqrt().cpu().numpy()
            )
            self._output_norms[position[0], position[1]] = (
                future_output.detach().float().square().sum(dim=(-2, -1)).sqrt().cpu().numpy()
            )
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
                raise RuntimeError("GEN block-residual collector captured no transformer outputs")
            expected_positions = self.expected_num_calls * len(self.layers)
            if self._pending_inputs:
                raise RuntimeError(f"Unmatched GEN block inputs remain: {sorted(self._pending_inputs)}")
            if (
                self._forward_call_index != self.expected_num_calls
                or len(self._written_positions) != expected_positions
            ):
                raise RuntimeError(
                    "Incomplete GEN block-residual capture: "
                    f"calls={self._forward_call_index}/{self.expected_num_calls}, "
                    f"call-blocks={len(self._written_positions)}/{expected_positions}"
                )
            if not np.isfinite(self._residual_norms).all() or not np.isfinite(self._output_norms).all():
                raise RuntimeError("NaN/Inf detected in captured residual or block-output norms")
            ratio = self._residual_norms / np.maximum(self._output_norms, np.finfo(np.float64).eps)
            np.save(self.output_dir / RESIDUAL_NORM_FILE, self._residual_norms)
            np.save(self.output_dir / OUTPUT_NORM_FILE, self._output_norms)
            np.save(self.output_dir / RESIDUAL_OUTPUT_RATIO_FILE, ratio)
            profile = {
                "raw_file": GEN_BLOCK_RESIDUAL_RAW_FILE,
                "raw_shape": list(self._raw_shape),
                "raw_dtype": str(self._raw_dtype),
                "raw_num_bytes": self._raw_num_bytes,
                "axis_names": [
                    "forward_call",
                    "transformer_block",
                    "future_latent",
                    "spatial_token",
                    "hidden_channel",
                ],
                "future_latent_labels": [f"L{index}" for index in range(1, self._latent_shape_thw[0])],
                "num_forward_calls": self.expected_num_calls,
                "num_layers": len(self.layers),
                "latent_shape_thw": list(self._latent_shape_thw),
                "hidden_size": self._hidden_size,
                "cfg_branch_by_call": [
                    "conditional" if self.guidance == 1.0 or index % 2 == 0 else "unconditional"
                    for index in range(self.expected_num_calls)
                ],
                "sampler_step_by_call": [
                    index if self.guidance == 1.0 else index // 2 for index in range(self.expected_num_calls)
                ],
                "timestep_by_call": [self._timesteps_by_call[index] for index in range(self.expected_num_calls)],
                "norm_files": {
                    "residual_frobenius": RESIDUAL_NORM_FILE,
                    "block_output_frobenius": OUTPUT_NORM_FILE,
                    "residual_to_output_ratio": RESIDUAL_OUTPUT_RATIO_FILE,
                },
            }
            self._raw_tensor = None
            with self.raw_path.open("rb") as handle:
                os.fsync(handle.fileno())
            return profile
        except Exception:
            self.cleanup()
            raise


def save_block_residual_capture_artifact(
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
    """Save one selected residual-capture action chunk and its comparison outputs."""

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
    profile_summary = {
        "schema_version": 1,
        "scope": "Exact GEN future-vision block residual R=H_out-H_in for every block and sampler call",
        **dict(profile),
        "storage_order": "C contiguous",
        "future_scope": "L1..L8 only; conditioning latent L0, text, and action tokens are excluded.",
        "spatial_token_rule": (
            f"spatial_token = patch_y * patch_width + patch_x; patch_height={patch_height}, patch_width={patch_width}"
        ),
        "capture_rule": (
            "A block pre-hook clones exact future-token H_in; its post-hook stores H_out-H_in. "
            "sampler_step_by_call, timestep_by_call, and cfg_branch_by_call preserve every path separately."
        ),
        "files": {
            "raw": GEN_BLOCK_RESIDUAL_RAW_FILE,
            "summary": GEN_BLOCK_RESIDUAL_PROFILE_FILE,
            **dict(profile["norm_files"]),
        },
    }
    (output_dir / GEN_BLOCK_RESIDUAL_PROFILE_FILE).write_text(
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
        "decoded_video_shape": list(pred_video.shape),
        "conditioning_observation_file": "conditioning_observation.png",
        "denoised_latent_file": "denoised_vision_latent.pt",
        "future_latent_file": "future_vision_latent.pt",
        "predicted_future_frames_dir": "predicted_future_frames",
        "gen_block_residual_profile": profile_summary,
        "completed_utc": _utc_now(),
    }
    (output_dir / CAPTURE_METADATA_FILE).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata
