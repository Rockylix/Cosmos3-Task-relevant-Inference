# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.scripts.robolab_attention_stats_capture import (
    GenAttentionStatsCollector,
    compute_group_attention_statistics,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _layout() -> dict:
    return {
        "num_gen_tokens": 20,
        "latent_shape_thw": [9, 1, 2],
        "latent_positions": {f"L{latent}": [2 * latent, 2 * latent + 1] for latent in range(9)},
        "action_positions": [18, 19],
    }


def test_group_stats_uniform_attention_has_baseline_mass_unit_enrichment_and_entropy() -> None:
    q = torch.zeros((20, 2, 4), dtype=torch.float32)
    k_ar = torch.zeros((3, 1, 4), dtype=torch.float32)
    k_gen = torch.zeros((20, 1, 4), dtype=torch.float32)
    group_rows, entropy_rows, value_rows, validation = compute_group_attention_statistics(
        torch=torch,
        q_gen=q,
        k_ar=k_ar,
        k_gen=k_gen,
        v_ar=torch.ones_like(k_ar),
        v_gen=torch.ones_like(k_gen),
        attn_output_gen=torch.ones_like(q),
        scaling=0.5,
        token_layout=_layout(),
        query_chunk_size=1,
    )

    assert len(group_rows) == 8 * 2 * 11
    assert len(entropy_rows) == 8 * 2
    assert len(value_rows) == 2 * 11
    for row in group_rows:
        assert row["mass_mean"] == pytest.approx(row["baseline_fraction"], abs=1e-7)
        assert row["enrichment_mean"] == pytest.approx(1.0, abs=1e-6)
        assert row["log2_enrichment_mean"] == pytest.approx(0.0, abs=1e-6)
        assert row["value_weighted_share_mean"] == pytest.approx(row["baseline_fraction"], abs=1e-7)
        assert row["value_weighted_enrichment_mean"] == pytest.approx(1.0, abs=1e-6)
        assert row["value_reweight_ratio_mean"] == pytest.approx(1.0, abs=1e-6)
        assert row["group_output_norm_mean"] == pytest.approx(2.0 * row["baseline_fraction"], abs=1e-6)
        assert row["cancellation_ratio_mean"] == pytest.approx(1.0, abs=1e-6)
        assert row["direction_contribution_mean"] == pytest.approx(row["baseline_fraction"], abs=1e-6)
    for row in entropy_rows:
        assert row["entropy_mean"] == pytest.approx(1.0, abs=1e-6)
    assert validation["group_partition_ok"]
    assert validation["finite_ok"]
    assert validation["mass_sum_error_max"] < 1e-6
    assert validation["dense_reference_error_max"] < 1e-7
    assert validation["gqa_repeat_factor"] == 2
    assert validation["value_share_sum_error_max"] < 1e-6
    assert validation["alpha_sum_error_max"] < 1e-6
    assert validation["cancellation_ratio_min"] == pytest.approx(1.0, abs=1e-6)
    assert validation["cancellation_ratio_max"] == pytest.approx(1.0, abs=1e-6)
    assert validation["actual_output_relative_l2_max"] < 1e-6


def test_value_statistics_use_consecutive_real_gqa_head_mapping() -> None:
    q = torch.zeros((20, 4, 4), dtype=torch.float32)
    k_ar = torch.zeros((3, 2, 4), dtype=torch.float32)
    k_gen = torch.zeros((20, 2, 4), dtype=torch.float32)
    v_ar = torch.zeros_like(k_ar)
    v_gen = torch.zeros_like(k_gen)
    # KV head 0 has L2 norm 2, KV head 1 has L2 norm 6. Consecutive Q
    # heads must map [0,0,1,1], exactly matching repeat_interleave GQA.
    v_ar[:, 0, :] = 1.0
    v_ar[:, 1, :] = 3.0
    v_gen[:, 0, :] = 1.0
    v_gen[:, 1, :] = 3.0
    actual_output = torch.empty_like(q)
    actual_output[:, :2, :] = 1.0
    actual_output[:, 2:, :] = 3.0

    group_rows, _entropy_rows, value_rows, validation = compute_group_attention_statistics(
        torch=torch,
        q_gen=q,
        k_ar=k_ar,
        k_gen=k_gen,
        v_ar=v_ar,
        v_gen=v_gen,
        attn_output_gen=actual_output,
        scaling=0.5,
        token_layout=_layout(),
        query_chunk_size=2,
    )

    assert validation["gqa_repeat_factor"] == 2
    by_head = {head: [row for row in value_rows if row["head"] == head] for head in range(4)}
    assert {row["kv_head"] for row in by_head[0] + by_head[1]} == {0}
    assert {row["kv_head"] for row in by_head[2] + by_head[3]} == {1}
    assert all(row["value_l2_mean"] == pytest.approx(2.0) for row in by_head[0] + by_head[1])
    assert all(row["value_l2_mean"] == pytest.approx(6.0) for row in by_head[2] + by_head[3])
    # Uniform QK and constant within-head V scale must leave the group share
    # unchanged after V weighting, despite the two KV heads having different scale.
    assert all(
        row["value_reweight_ratio_mean"] == pytest.approx(1.0, abs=1e-6) for row in group_rows
    )


def test_group_stats_rejects_incomplete_gen_partition() -> None:
    layout = _layout()
    layout["action_positions"] = [18]
    with pytest.raises(RuntimeError, match="partition actual attention keys"):
        compute_group_attention_statistics(
            torch=torch,
            q_gen=torch.zeros((20, 2, 4)),
            k_ar=torch.zeros((3, 1, 4)),
            k_gen=torch.zeros((20, 1, 4)),
            v_ar=torch.ones((3, 1, 4)),
            v_gen=torch.ones((20, 1, 4)),
            attn_output_gen=torch.ones((20, 2, 4)),
            scaling=0.5,
            token_layout=layout,
        )


class _FakeAttention:
    def __init__(self, layer_index: int) -> None:
        self.layer_index = layer_index
        self._attention_stats_capture_callback = None

    def run(self, num_gen_tokens: int, num_ar_tokens: int) -> None:
        callback = self._attention_stats_capture_callback
        if callback is None:
            return
        callback(
            layer_index=self.layer_index,
            q_gen=torch.zeros((num_gen_tokens, 2, 4)),
            k_ar=torch.zeros((num_ar_tokens, 1, 4)),
            k_gen=torch.zeros((num_gen_tokens, 1, 4)),
            v_ar=torch.ones((num_ar_tokens, 1, 4)),
            v_gen=torch.ones((num_gen_tokens, 1, 4)),
            attn_output_gen=torch.ones((num_gen_tokens, 2, 4)),
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

    def forward(self, packed_sequence: SimpleNamespace, memory: object = None, und_only: bool = False) -> torch.Tensor:
        del memory
        if not und_only:
            for layer in self.layers:
                layer.self_attn.run(num_gen_tokens=20, num_ar_tokens=3)
        # The experiment callbacks must not alter the forward result.
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


def test_collector_step_branch_block_gating_and_read_only_output(tmp_path: Path) -> None:
    net = _FakeNet()
    packed = _packed(999.0)
    baseline = net(packed).clone()
    collector = GenAttentionStatsCollector(
        torch=torch,
        net=net,
        output_dir=tmp_path / "stats",
        guidance=3.0,
        num_steps=2,
        shift=5.0,
        task="BananaInBowlTask",
        chunk=3,
        query_chunk_size=1,
    )
    outputs = []
    with collector:
        for call in range(4):
            packed.vision.timesteps.fill_(999.0 - 100.0 * (call // 2))
            outputs.append(net(packed))
    profile = collector.finish()

    assert all(torch.equal(output, baseline) for output in outputs)
    assert profile["validation"]["call_block_count"] == 2 * 2 * 2
    assert [item["timestep"] for item in profile["timesteps"]] == [999.0, 999.0, 899.0, 899.0]
    with (tmp_path / "stats" / "calls.csv").open(newline="", encoding="utf-8") as handle:
        calls = list(csv.DictReader(handle))
    assert len(calls) == 8
    assert {(row["step"], row["branch"], row["block"]) for row in calls} == {
        (str(step), branch, str(block))
        for step in range(2)
        for branch in ("conditional", "unconditional")
        for block in range(2)
    }
    validation = json.loads((tmp_path / "stats" / "validation.json").read_text(encoding="utf-8"))
    assert validation["validation"]["mass_sum_error_max"] < 1e-6


def test_collector_can_gate_to_one_step_branch_and_block(tmp_path: Path) -> None:
    net = _FakeNet()
    collector = GenAttentionStatsCollector(
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
        query_chunk_size=2,
    )
    with collector:
        for call in range(4):
            net(_packed(999.0 - 100.0 * (call // 2)))
    profile = collector.finish()
    assert profile["validation"]["call_block_count"] == 1
    assert profile["selected_steps"] == [1]
    assert profile["selected_blocks"] == [1]
    assert profile["selected_branches"] == ["conditional"]
