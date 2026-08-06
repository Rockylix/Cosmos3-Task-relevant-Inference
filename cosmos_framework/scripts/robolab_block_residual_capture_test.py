# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos_framework.scripts.robolab_block_residual_capture import (
    GenBlockResidualCollector,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class _FakeLayer(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(
        self,
        pack: dict[str, torch.Tensor],
        **kwargs: object,
    ) -> tuple[dict[str, torch.Tensor], dict, None]:
        del kwargs
        output = dict(pack)
        output["full_only_seq"] = pack["full_only_seq"] * self.scale
        return output, {}, None


class _FakeNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers = torch.nn.ModuleList([_FakeLayer(2.0), _FakeLayer(3.0)])
        self.language_model = SimpleNamespace(model=SimpleNamespace(layers=layers))

    def forward(
        self,
        packed_sequence: SimpleNamespace,
        memory: object = None,
        und_only: bool = False,
    ) -> dict:
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


def test_collector_uses_exact_pre_post_and_excludes_l0(tmp_path: Path) -> None:
    # Three latent frames, two tokens per frame. L0 is intentionally very large
    # so any accidental inclusion is immediately visible.
    hidden = torch.tensor(
        [
            [100.0, 0.0],
            [100.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
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
        full_hidden=hidden,
    )
    net = _FakeNet()
    collector = GenBlockResidualCollector(
        torch=torch,
        net=net,
        guidance=3.0,
        num_steps=1,
        output_dir=tmp_path,
        disk_reserve_bytes=0,
    )
    with collector:
        net(packed)
        net(packed)
    profile = collector.finish()

    assert profile["raw_shape"] == [2, 2, 2, 2, 2]
    assert profile["future_latent_labels"] == ["L1", "L2"]
    assert profile["sampler_step_by_call"] == [0, 0]
    assert profile["cfg_branch_by_call"] == ["conditional", "unconditional"]
    raw = torch.from_file(
        str(tmp_path / "gen_block_residual_raw.bin"),
        shared=False,
        size=2 * 2 * 2 * 2 * 2,
        dtype=torch.bfloat16,
    ).reshape(2, 2, 2, 2, 2)
    future = hidden[2:].reshape(2, 2, 2)
    torch.testing.assert_close(raw[0, 0], future)
    torch.testing.assert_close(raw[0, 1], future * 4.0)
    assert float(raw.abs().max()) < 100.0

    residual_norms = np.load(tmp_path / "residual_frobenius_norm.npy")
    output_norms = np.load(tmp_path / "block_output_frobenius_norm.npy")
    ratios = np.load(tmp_path / "residual_to_output_ratio.npy")
    np.testing.assert_allclose(ratios, residual_norms / output_norms)
    np.testing.assert_allclose(ratios[:, 0], 0.5)
    np.testing.assert_allclose(ratios[:, 1], 2.0 / 3.0)


def test_collector_ignores_und_only_forward(tmp_path: Path) -> None:
    hidden = torch.ones((4, 2), dtype=torch.bfloat16)
    packed = SimpleNamespace(
        vision=SimpleNamespace(
            token_shapes=[(2, 1, 2)],
            sequence_indexes=torch.arange(0, 4),
            timesteps=torch.ones(4),
        ),
        attn_modes=["full"],
        split_lens=[4],
        full_hidden=hidden,
    )
    net = _FakeNet()
    collector = GenBlockResidualCollector(
        torch=torch,
        net=net,
        guidance=1.0,
        num_steps=1,
        output_dir=tmp_path,
        disk_reserve_bytes=0,
    )
    with collector:
        net(packed, und_only=True)
        net(packed)
    profile = collector.finish()
    assert profile["num_forward_calls"] == 1
