# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.scripts.robolab_spatial_token_similarity_capture import (
    REPRESENTATIONS,
    SpatialTokenSimilarityCollector,
    compute_adjacent_spatial_cosine,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _layout() -> dict:
    return {
        "num_gen_tokens": 20,
        "latent_shape_thw": [9, 1, 2],
        "latent_positions": {f"L{latent}": [2 * latent, 2 * latent + 1] for latent in range(9)},
        "action_positions": [18, 19],
    }


def test_spatial_cosine_matches_same_coordinate_and_averages_seven_pairs() -> None:
    tensor = torch.zeros((20, 2), dtype=torch.float32)
    # Coordinate 0 stays collinear across all future latents: cosine 1.
    for latent in range(1, 9):
        tensor[2 * latent] = torch.tensor([float(latent), 0.0])
    # Coordinate 1 alternates orthogonal axes: every adjacent cosine 0.
    for latent in range(1, 9):
        tensor[2 * latent + 1] = torch.tensor([1.0, 0.0] if latent % 2 else [0.0, 1.0])

    pair_maps, mean_map, validation = compute_adjacent_spatial_cosine(
        torch=torch, tensor=tensor, token_layout=_layout()
    )

    assert pair_maps.shape == (7, 1, 2)
    assert torch.allclose(pair_maps[:, 0, 0], torch.ones(7))
    assert torch.allclose(pair_maps[:, 0, 1], torch.zeros(7))
    assert torch.allclose(mean_map, torch.tensor([[1.0, 0.0]]))
    assert validation["flattened_feature_dim"] == 2
    assert validation["finite"]


def test_spatial_cosine_flattens_all_head_and_head_dim_axes() -> None:
    tensor = torch.zeros((20, 2, 3), dtype=torch.float32)
    for latent in range(1, 9):
        tensor[2 * latent] = 1.0
        tensor[2 * latent + 1] = -1.0 if latent % 2 else 1.0
    pair_maps, _mean_map, validation = compute_adjacent_spatial_cosine(
        torch=torch, tensor=tensor, token_layout=_layout()
    )
    assert validation["flattened_feature_dim"] == 6
    assert torch.allclose(pair_maps[:, 0, 0], torch.ones(7))
    assert torch.allclose(pair_maps[:, 0, 1], -torch.ones(7))


class _FakeAttention:
    def __init__(self, layer_index: int) -> None:
        self.layer_index = layer_index
        self._attention_stats_capture_callback = None
        self._rope_qk_capture_callback = None

    def run(self, tensor: torch.Tensor) -> None:
        pre_rope = tensor.reshape(20, 1, 2)
        post_rope = pre_rope.clone()
        # Make post-RoPE adjacent latents point in the opposite direction so
        # the artifact test can prove that Q/K came from the pre-RoPE hook.
        for latent in range(9):
            post_rope[2 * latent : 2 * latent + 2] *= -1.0 if latent % 2 else 1.0
        if self._rope_qk_capture_callback is not None:
            self._rope_qk_capture_callback(
                layer_index=self.layer_index,
                q_raw=pre_rope,
                k_raw=pre_rope,
                q_rope=post_rope,
                k_rope=post_rope,
                cos=torch.ones((20, 2)),
                sin=torch.zeros((20, 2)),
            )
        if self._attention_stats_capture_callback is not None:
            self._attention_stats_capture_callback(
                layer_index=self.layer_index,
                q_gen=post_rope,
                k_ar=torch.zeros((3, 1, 2)),
                k_gen=post_rope,
                v_ar=torch.zeros((3, 1, 2)),
                v_gen=pre_rope,
                attn_output_gen=pre_rope,
                scaling=1.0,
            )


class _FakeLayer(torch.nn.Module):
    def __init__(self, layer_index: int) -> None:
        super().__init__()
        self.self_attn = _FakeAttention(layer_index)
        self.mlp_moe_gen = torch.nn.Identity()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        self.self_attn.run(tensor)
        return self.mlp_moe_gen(tensor)


class _FakeNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_FakeLayer(0), _FakeLayer(1)])
        self.language_model = SimpleNamespace(model=SimpleNamespace(layers=self.layers))

    def forward(self, packed_sequence: SimpleNamespace, memory: object = None, und_only: bool = False) -> torch.Tensor:
        del memory
        tensor = torch.arange(40, dtype=torch.float32).reshape(20, 2) + 1.0
        if not und_only:
            for layer in self.layers:
                tensor = layer(tensor)
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


def test_collector_captures_five_representations_with_step_branch_gating(tmp_path: Path) -> None:
    net = _FakeNet()
    packed = _packed(999.0)
    baseline = net(packed).clone()
    collector = SpatialTokenSimilarityCollector(
        torch=torch,
        net=net,
        output_dir=tmp_path / "spatial",
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
    assert profile["validation"]["finite"]
    assert profile["validation"]["map_count"] == len(REPRESENTATIONS) * 2 * 2 * 2
    with (tmp_path / "spatial" / "spatial_similarity.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(REPRESENTATIONS) * 2 * 2 * 2 * 2
    assert {row["representation"] for row in rows} == set(REPRESENTATIONS)
    assert {row["branch"] for row in rows} == {"conditional", "unconditional"}
    assert profile["schema_version"] == 2
    assert "pre RoPE" in profile["boundaries"]["Q"]
    assert "pre RoPE" in profile["boundaries"]["K"]
    artifacts = torch.load(
        tmp_path / "spatial" / "spatial_similarity_maps.pt",
        map_location="cpu",
        weights_only=False,
    )
    for representation in ("Q", "K"):
        pair_maps = artifacts[(representation, 0, "conditional", 0)]["pair_maps"]
        assert bool((pair_maps > 0.0).all())


def test_collector_can_gate_one_step_branch_and_block(tmp_path: Path) -> None:
    net = _FakeNet()
    collector = SpatialTokenSimilarityCollector(
        torch=torch,
        net=net,
        output_dir=tmp_path / "gated",
        guidance=3.0,
        num_steps=2,
        shift=5.0,
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
    assert profile["validation"]["map_count"] == len(REPRESENTATIONS)
    assert profile["selected_steps"] == [1]
    assert profile["selected_blocks"] == [1]
    assert profile["selected_branches"] == ["conditional"]
