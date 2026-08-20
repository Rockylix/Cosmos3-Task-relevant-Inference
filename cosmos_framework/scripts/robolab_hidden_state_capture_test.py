# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos_framework.scripts.robolab_hidden_state_capture import (
    GenHiddenStateCollector,
    HiddenStateCapturePlanner,
    save_capture_artifact,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_planner_counts_chunks_independently_per_prompt(tmp_path: Path) -> None:
    planner = HiddenStateCapturePlanner(
        output_root=tmp_path / "experiment",
        chunk_indices=[1, 3],
        experiment_metadata={"checkpoint_path": "/model"},
    )

    assert planner.select(prompt="task a", seed=1) is None
    task_a_chunk_1 = planner.select(prompt="task a", seed=2)
    assert task_a_chunk_1 is not None
    assert task_a_chunk_1.chunk_index == 1

    assert planner.select(prompt="task b", seed=3) is None
    task_b_chunk_1 = planner.select(prompt="task b", seed=4)
    assert task_b_chunk_1 is not None
    assert task_b_chunk_1.chunk_index == 1
    assert task_a_chunk_1.task_key != task_b_chunk_1.task_key


class _FakeLayer(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, pack: dict[str, torch.Tensor], **kwargs: object) -> tuple[dict[str, torch.Tensor], dict, None]:
        del kwargs
        output = dict(pack)
        output["full_only_seq"] = pack["full_only_seq"] * self.scale
        return output, {}, None


class _FakeNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers = torch.nn.ModuleList([_FakeLayer(1.0), _FakeLayer(2.0)])
        self.language_model = SimpleNamespace(model=SimpleNamespace(layers=layers))
        self.config = SimpleNamespace(temporal_compression_factor_vision=4)

    def forward(self, packed_sequence: SimpleNamespace, memory: object = None, und_only: bool = False) -> dict:
        del memory
        pack = {
            "full_only_seq": packed_sequence.full_hidden.clone(),
            "_num_full_tokens": packed_sequence.full_hidden.shape[0],
        }
        if und_only:
            return pack
        for layer in self.language_model.model.layers:
            pack, _, _ = layer(pack, gen_only=True, und_only=False)
        return pack


def test_collector_and_artifact_preserve_five_axes_and_future_frames(tmp_path: Path) -> None:
    vision_hidden = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [4.0, 0.0],
        ],
        dtype=torch.bfloat16,
    )
    packed = SimpleNamespace(
        vision=SimpleNamespace(
            token_shapes=[(3, 1, 2)],
            sequence_indexes=torch.arange(0, 6),
            timesteps=torch.full((4,), 0.5),
        ),
        attn_modes=["full"],
        split_lens=[6],
        full_hidden=vision_hidden,
    )
    planner = HiddenStateCapturePlanner(
        output_root=tmp_path / "experiment",
        chunk_indices=[0],
        experiment_metadata={},
    )
    selection = planner.select(prompt="pick object", seed=7)
    assert selection is not None
    net = _FakeNet()
    collector = GenHiddenStateCollector(
        torch=torch,
        net=net,
        guidance=3.0,
        num_steps=1,
        output_dir=selection.partial_dir,
        disk_reserve_bytes=0,
    )
    with collector:
        net(packed)
        net(packed)
    profile = collector.finish()

    assert profile["raw_shape"] == [2, 2, 3, 2, 2]
    assert profile["sampler_step_by_call"] == [0, 0]
    assert profile["cfg_branch_by_call"] == ["conditional", "unconditional"]
    raw = torch.from_file(
        str(selection.partial_dir / "gen_hidden_state_raw.bin"),
        shared=False,
        size=2 * 2 * 3 * 2 * 2,
        dtype=torch.bfloat16,
    ).reshape(2, 2, 3, 2, 2)
    torch.testing.assert_close(raw[0, 0], vision_hidden.reshape(3, 2, 2))
    torch.testing.assert_close(raw[0, 1], vision_hidden.reshape(3, 2, 2) * 2)

    latent = torch.zeros((1, 4, 3, 2, 2), dtype=torch.float32)
    pred_video = torch.linspace(-1.0, 1.0, 3 * 9 * 2 * 2).reshape(1, 3, 9, 2, 2)
    metadata = save_capture_artifact(
        torch=torch,
        selection=selection,
        profile=profile,
        vision_latent=latent,
        pred_video=pred_video,
        conditioning_image=np.zeros((4, 5, 3), dtype=np.uint8),
        fps=15.0,
        action_chunk_size=32,
    )
    planner.complete(selection, metadata)

    assert selection.final_dir.is_dir()
    assert not selection.partial_dir.exists()
    assert len(list((selection.final_dir / "predicted_future_frames").glob("frame_*.png"))) == 8
    assert metadata["gen_hidden_state_profile"]["decoded_frame_ranges_by_latent_frame"] == [
        [0, 0],
        [1, 4],
        [5, 8],
    ]
    manifest = json.loads((planner.output_root / "manifest.jsonl").read_text(encoding="utf-8"))
    assert manifest["raw_shape"] == [2, 2, 3, 2, 2]
