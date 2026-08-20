# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos_framework.scripts.robolab_rope_qk_capture import (
    GenRopeQKCollector,
    RopeQKCapturePlanner,
    save_rope_qk_capture_artifact,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class _FakeAttention:
    def __init__(self, layer_index: int) -> None:
        self.layer_index = layer_index
        self._rope_qk_capture_callback = None

    def run(self, *, call_index: int, num_tokens: int) -> None:
        callback = self._rope_qk_capture_callback
        if callback is None:
            return
        base = torch.arange(num_tokens * 2 * 2, dtype=torch.bfloat16).reshape(num_tokens, 2, 2)
        q_raw = base + 10 * call_index + self.layer_index
        k_raw = base[:, :1] + 20 * call_index + self.layer_index
        callback(
            layer_index=self.layer_index,
            q_raw=q_raw,
            k_raw=k_raw,
            q_rope=q_raw.flip(-1),
            k_rope=k_raw.flip(-1),
            cos=torch.ones((num_tokens, 2), dtype=torch.bfloat16),
            sin=torch.zeros((num_tokens, 2), dtype=torch.bfloat16),
        )


class _FakeLayer:
    def __init__(self, layer_index: int) -> None:
        self.self_attn = _FakeAttention(layer_index)


class _FakeRopeNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = [_FakeLayer(0), _FakeLayer(1)]
        self.language_model = SimpleNamespace(model=SimpleNamespace(layers=self.layers))
        self.call_index = 0

    def forward(self, packed_sequence: SimpleNamespace, memory: object = None, und_only: bool = False) -> dict:
        del memory
        if not und_only:
            num_tokens = sum(
                int(length)
                for mode, length in zip(
                    packed_sequence.attn_modes,
                    packed_sequence.split_lens,
                    strict=True,
                )
                if mode == "full"
            )
            for layer in self.layers:
                layer.self_attn.run(call_index=self.call_index, num_tokens=num_tokens)
            self.call_index += 1
        return {}


def _make_packed() -> SimpleNamespace:
    return SimpleNamespace(
        vision=SimpleNamespace(
            token_shapes=[(3, 1, 2)],
            sequence_indexes=torch.arange(0, 6),
            timesteps=torch.full((6,), 0.5),
        ),
        action=SimpleNamespace(
            token_shapes=[(2,)],
            sequence_indexes=torch.arange(6, 8),
        ),
        attn_modes=["full"],
        split_lens=[8],
        position_ids=torch.stack(
            [
                torch.arange(8),
                torch.arange(8) % 2,
                torch.arange(8) % 3,
            ]
        ),
    )


def test_rope_qk_collector_selects_steps_blocks_and_conditional_branch(tmp_path: Path) -> None:
    planner = RopeQKCapturePlanner(
        output_root=tmp_path / "experiment",
        chunk_indices=[0],
        experiment_metadata={},
    )
    selection = planner.select(prompt="pick object", seed=7)
    assert selection is not None
    net = _FakeRopeNet()
    collector = GenRopeQKCollector(
        torch=torch,
        net=net,
        guidance=3.0,
        num_steps=2,
        selected_steps=[0, 1],
        selected_blocks=[0, 1],
        selected_branches=["conditional"],
        output_dir=selection.partial_dir,
        disk_reserve_bytes=0,
    )
    packed = _make_packed()
    with collector:
        for _ in range(4):
            net(packed)
    profile = collector.finish()

    assert profile["raw_shapes"]["q_raw"] == [2, 2, 8, 2, 2]
    assert profile["raw_shapes"]["k_raw"] == [2, 2, 8, 1, 2]
    assert [call["forward_call_index"] for call in profile["selected_calls"]] == [0, 2]
    assert profile["token_layout"]["vision"]["gen_position_ranges"] == [[0, 6]]
    assert profile["token_layout"]["action"]["gen_position_ranges"] == [[6, 8]]

    q_raw = torch.from_file(
        str(selection.partial_dir / "q_raw.bin"),
        shared=False,
        size=2 * 2 * 8 * 2 * 2,
        dtype=torch.bfloat16,
    ).reshape(2, 2, 8, 2, 2)
    q_rope = torch.from_file(
        str(selection.partial_dir / "q_rope.bin"),
        shared=False,
        size=2 * 2 * 8 * 2 * 2,
        dtype=torch.bfloat16,
    ).reshape(2, 2, 8, 2, 2)
    torch.testing.assert_close(q_rope, q_raw.flip(-1))

    position_payload = torch.load(selection.partial_dir / "position_ids.pt", weights_only=True)
    assert tuple(position_payload["position_ids"].shape) == (2, 3, 8)

    latent = torch.zeros((1, 4, 3, 2, 2), dtype=torch.float32)
    pred_video = torch.linspace(-1.0, 1.0, 3 * 9 * 2 * 2).reshape(1, 3, 9, 2, 2)
    metadata = save_rope_qk_capture_artifact(
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
    assert len(list((selection.final_dir / "predicted_future_frames").glob("frame_*.png"))) == 8
    manifest = json.loads((planner.output_root / "manifest.jsonl").read_text(encoding="utf-8"))
    assert manifest["selected_steps"] == [0, 1]
    assert manifest["selected_blocks"] == [0, 1]
