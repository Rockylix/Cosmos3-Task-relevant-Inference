# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.scripts.robolab_action_query_attention_capture import (
    ActionQueryAttentionCollector,
    _action_query_layout,
    compute_action_query_statistics,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _layout() -> dict:
    return {
        "num_gen_tokens": 21,
        "latent_shape_thw": [9, 1, 2],
        "latent_positions": {f"L{latent}": [2 * latent, 2 * latent + 1] for latent in range(9)},
        "action_positions": [18, 19, 20],
        "action_token_shape": [3],
        "action_queries": [
            {
                "query_action_index": 0,
                "gen_position": 18,
                "query_role": "condition",
                "action_horizon": -1,
            },
            {
                "query_action_index": 1,
                "gen_position": 19,
                "query_role": "predicted",
                "action_horizon": 0,
            },
            {
                "query_action_index": 2,
                "gen_position": 20,
                "query_role": "predicted",
                "action_horizon": 1,
            },
        ],
    }


def test_uniform_action_query_attention_matches_group_baselines_and_true_output() -> None:
    q = torch.zeros((21, 2, 4), dtype=torch.float32)
    k_ar = torch.zeros((3, 1, 4), dtype=torch.float32)
    k_gen = torch.zeros((21, 1, 4), dtype=torch.float32)
    v_ar = torch.ones_like(k_ar)
    v_gen = torch.ones_like(k_gen)
    group_rows, entropy_rows, validation = compute_action_query_statistics(
        torch=torch,
        q_gen=q,
        k_ar=k_ar,
        k_gen=k_gen,
        v_ar=v_ar,
        v_gen=v_gen,
        attn_output_gen=torch.ones_like(q),
        scaling=0.5,
        token_layout=_layout(),
    )

    assert len(group_rows) == 3 * 2 * 11
    assert len(entropy_rows) == 3 * 2
    assert {row["query_role"] for row in group_rows} == {"condition", "predicted"}
    assert {row["action_horizon"] for row in group_rows} == {-1, 0, 1}
    for row in group_rows:
        assert row["mass"] == pytest.approx(row["baseline_fraction"], abs=1e-7)
        assert row["enrichment"] == pytest.approx(1.0, abs=1e-6)
        assert row["cancellation_ratio"] == pytest.approx(1.0, abs=1e-6)
        assert row["direction_contribution"] == pytest.approx(
            row["baseline_fraction"], abs=1e-6
        )
    assert all(row["normalized_entropy"] == pytest.approx(1.0) for row in entropy_rows)
    assert validation["group_partition_ok"]
    assert validation["finite"]
    assert validation["gqa_repeat_factor"] == 2
    assert validation["mass_sum_error_max"] < 1e-6
    assert validation["alpha_sum_error_max"] < 1e-6
    assert validation["actual_output_relative_l2"] < 1e-6
    assert validation["actual_output_cosine"] == pytest.approx(1.0)


def test_real_gqa_mapping_is_consecutive_for_action_query_outputs() -> None:
    q = torch.zeros((21, 4, 4), dtype=torch.float32)
    k_ar = torch.zeros((3, 2, 4), dtype=torch.float32)
    k_gen = torch.zeros((21, 2, 4), dtype=torch.float32)
    v_ar = torch.empty_like(k_ar)
    v_gen = torch.empty_like(k_gen)
    v_ar[:, 0] = 1.0
    v_ar[:, 1] = 3.0
    v_gen[:, 0] = 1.0
    v_gen[:, 1] = 3.0
    actual_output = torch.empty_like(q)
    actual_output[:, :2] = 1.0
    actual_output[:, 2:] = 3.0
    rows, _entropy, validation = compute_action_query_statistics(
        torch=torch,
        q_gen=q,
        k_ar=k_ar,
        k_gen=k_gen,
        v_ar=v_ar,
        v_gen=v_gen,
        attn_output_gen=actual_output,
        scaling=0.5,
        token_layout=_layout(),
    )
    assert validation["gqa_repeat_factor"] == 2
    assert {row["kv_head"] for row in rows if row["head"] in (0, 1)} == {0}
    assert {row["kv_head"] for row in rows if row["head"] in (2, 3)} == {1}
    assert validation["actual_output_relative_l2"] < 1e-6


def _packed(timestep: float) -> SimpleNamespace:
    return SimpleNamespace(
        vision=SimpleNamespace(
            token_shapes=[(9, 1, 2)],
            sequence_indexes=torch.arange(20, 38),
            timesteps=torch.full((18,), timestep),
        ),
        action=SimpleNamespace(
            token_shapes=[(3,)],
            sequence_indexes=torch.arange(38, 41),
            condition_mask=[torch.tensor([[1.0], [0.0], [0.0]])],
        ),
        attn_modes=["causal", "full"],
        split_lens=[20, 21],
    )


def test_action_layout_labels_condition_and_predicted_horizons() -> None:
    layout = _action_query_layout(torch, _packed(999.0))
    assert layout["action_positions"] == [18, 19, 20]
    assert [query["query_role"] for query in layout["action_queries"]] == [
        "condition",
        "predicted",
        "predicted",
    ]
    assert [query["action_horizon"] for query in layout["action_queries"]] == [-1, 0, 1]


class _FakeAttention:
    def __init__(self, layer_index: int) -> None:
        self.layer_index = layer_index
        self._attention_stats_capture_callback = None

    def run(self) -> None:
        callback = self._attention_stats_capture_callback
        if callback is None:
            return
        callback(
            layer_index=self.layer_index,
            q_gen=torch.zeros((21, 2, 4)),
            k_ar=torch.zeros((3, 1, 4)),
            k_gen=torch.zeros((21, 1, 4)),
            v_ar=torch.ones((3, 1, 4)),
            v_gen=torch.ones((21, 1, 4)),
            attn_output_gen=torch.ones((21, 2, 4)),
            scaling=0.5,
        )


class _FakeLayer:
    def __init__(self, layer_index: int) -> None:
        self.self_attn = _FakeAttention(layer_index)


class _FakeNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = [_FakeLayer(0), _FakeLayer(1)]
        self.language_model = SimpleNamespace(model=SimpleNamespace(layers=self.layers))

    def forward(
        self, packed_sequence: SimpleNamespace, memory: object = None, und_only: bool = False
    ) -> torch.Tensor:
        del packed_sequence, memory
        if not und_only:
            for layer in self.layers:
                layer.self_attn.run()
        return torch.tensor(17.0)


def test_collector_step_branch_block_gating_and_read_only_output(tmp_path: Path) -> None:
    net = _FakeNet()
    packed = _packed(999.0)
    baseline = net(packed).clone()
    collector = ActionQueryAttentionCollector(
        torch=torch,
        net=net,
        output_dir=tmp_path / "action_query",
        guidance=3.0,
        num_steps=2,
        shift=5.0,
        task="task",
        chunk=3,
        selected_blocks=[0, 1],
    )
    outputs = []
    with collector:
        for call in range(4):
            packed.vision.timesteps.fill_(999.0 - 100.0 * (call // 2))
            outputs.append(net(packed))
    profile = collector.finish()

    assert all(torch.equal(output, baseline) for output in outputs)
    assert profile["validation"]["complete"]
    assert profile["validation"]["call_block_count"] == 2 * 2 * 2
    assert profile["validation"]["num_action_queries"] == 3
    with (tmp_path / "action_query" / "calls.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        calls = list(csv.DictReader(handle))
    assert len(calls) == 8
    assert {(row["step"], row["branch"], row["block"]) for row in calls} == {
        (str(step), branch, str(block))
        for step in range(2)
        for branch in ("conditional", "unconditional")
        for block in range(2)
    }


def test_collector_can_gate_one_step_branch_and_block(tmp_path: Path) -> None:
    net = _FakeNet()
    collector = ActionQueryAttentionCollector(
        torch=torch,
        net=net,
        output_dir=tmp_path / "gated",
        guidance=3.0,
        num_steps=2,
        shift=1.0,
        task="task",
        chunk=3,
        selected_steps=[1],
        selected_blocks=[1],
        selected_branches=["conditional"],
    )
    with collector:
        for call in range(4):
            net(_packed(999.0 - 100.0 * (call // 2)))
    profile = collector.finish()
    assert profile["validation"]["call_block_count"] == 1
    assert profile["selected_steps"] == [1]
    assert profile["selected_blocks"] == [1]
    assert profile["selected_branches"] == ["conditional"]
