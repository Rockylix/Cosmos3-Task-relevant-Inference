# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.scripts.robolab_qk_reuse_profile_capture import (
    QKReuseProfileCollector,
    compute_qk_reuse_profile,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _layout() -> dict:
    return {
        "num_gen_tokens": 20,
        "latent_shape_thw": [9, 1, 2],
        "latent_positions": {f"L{latent}": [2 * latent, 2 * latent + 1] for latent in range(9)},
        "action_positions": [18, 19],
    }


def test_profile_detects_joint_qk_candidates_and_preserves_pair_order() -> None:
    q = torch.zeros((20, 2, 2), dtype=torch.float32)
    k = torch.zeros((20, 1, 2), dtype=torch.float32)
    for latent in range(1, 9):
        q[2 * latent : 2 * latent + 2] = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        k[2 * latent : 2 * latent + 2] = torch.tensor([[[1.0, 0.0]]])
    # Make only L7 differ so pair L6->L7 and L7->L8 fail while earlier pairs pass.
    q[14:16] = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    k[14:16] = torch.tensor([[[0.0, 1.0]]])
    rows = compute_qk_reuse_profile(
        torch=torch, q_gen=q, k_gen=k, token_layout=_layout(), threshold=0.9
    )
    assert [row["future_pair"] for row in rows] == [f"L{i}_L{i+1}" for i in range(1, 8)]
    assert all(row["candidate_full_cosine"] for row in rows[:5])
    assert not rows[5]["candidate_full_cosine"]
    assert not rows[6]["candidate_full_cosine"]
    assert rows[0]["q_token_fraction_ge_threshold"] == pytest.approx(1.0)


def test_full_cosine_and_token_mean_are_both_reported() -> None:
    q = torch.ones((20, 2, 2), dtype=torch.float32)
    k = torch.ones((20, 1, 2), dtype=torch.float32)
    rows = compute_qk_reuse_profile(
        torch=torch, q_gen=q, k_gen=k, token_layout=_layout(), threshold=0.9
    )
    for row in rows:
        assert row["q_full_cosine"] == pytest.approx(1.0)
        assert row["k_full_cosine"] == pytest.approx(1.0)
        assert row["joint_full_cosine_min"] == pytest.approx(1.0)
        assert row["joint_token_cosine_mean_min"] == pytest.approx(1.0)
        assert row["candidate_token_p10"]


class _FakeAttention:
    def __init__(self, layer_index: int) -> None:
        self.layer_index = layer_index
        self._attention_stats_capture_callback = None

    def run(self) -> None:
        if self._attention_stats_capture_callback is None:
            return
        self._attention_stats_capture_callback(
            layer_index=self.layer_index,
            q_gen=torch.ones((20, 2, 2)),
            k_ar=torch.zeros((3, 1, 2)),
            k_gen=torch.ones((20, 1, 2)),
            v_ar=torch.zeros((3, 1, 2)),
            v_gen=torch.ones((20, 1, 2)),
            attn_output_gen=torch.ones((20, 2, 2)),
            scaling=1.0,
        )


class _FakeLayer:
    def __init__(self, layer_index: int) -> None:
        self.self_attn = _FakeAttention(layer_index)


class _FakeNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = [_FakeLayer(0), _FakeLayer(1)]
        self.language_model = SimpleNamespace(model=SimpleNamespace(layers=self.layers))

    def forward(self, packed: SimpleNamespace, memory: object = None, und_only: bool = False) -> torch.Tensor:
        del packed, memory
        if not und_only:
            for layer in self.layers:
                layer.self_attn.run()
        return torch.tensor(17.0)


def _packed(timestep: float) -> SimpleNamespace:
    return SimpleNamespace(
        vision=SimpleNamespace(
            token_shapes=[(9, 1, 2)],
            sequence_indexes=torch.arange(20, 38),
            timesteps=torch.full((18,), timestep),
        ),
        action=SimpleNamespace(token_shapes=[(2,)], sequence_indexes=torch.arange(38, 40)),
        attn_modes=["causal", "full"],
        split_lens=[20, 20],
    )


def test_collector_profiles_steps_branches_blocks_and_is_read_only(tmp_path: Path) -> None:
    net = _FakeNet()
    collector = QKReuseProfileCollector(
        torch=torch,
        net=net,
        output_dir=tmp_path / "profile",
        guidance=3.0,
        num_steps=2,
        shift=5.0,
        task="task",
        chunk=3,
        threshold=0.9,
    )
    outputs = []
    with collector:
        for call in range(4):
            outputs.append(net(_packed(999.0 - 100.0 * (call // 2))))
    profile = collector.finish()
    assert all(output.item() == 17.0 for output in outputs)
    assert profile["validation"]["call_block_count"] == 2 * 2 * 2
    assert profile["validation"]["row_count"] == 2 * 2 * 2 * 7
    with (tmp_path / "profile" / "qk_adjacent_profile.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 56
    assert all(row["candidate_full_cosine"] == "True" for row in rows)
