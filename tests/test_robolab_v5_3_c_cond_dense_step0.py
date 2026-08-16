from __future__ import annotations

import pytest
import torch

from cosmos_framework.scripts.robolab_v5_3_c_cond_dense_step0 import (
    step0_unconditional_group,
    zero_cfg_delta_outside_future_roi,
)


def test_step0_unconditional_uses_g1_for_dense_head_then_normal_groups() -> None:
    groups = ((4, 11), (12, 19), (20, 27))
    assert [step0_unconditional_group(block, groups) for block in range(28)] == (
        [0] * 12 + [1] * 8 + [2] * 8
    )
    with pytest.raises(ValueError, match="outside"):
        step0_unconditional_group(28, groups)


@pytest.mark.parametrize("batched", [False, True])
def test_zero_cfg_delta_replaces_only_future_background(batched: bool) -> None:
    shape = (2, 9, 2, 3)
    conditional = torch.arange(torch.tensor(shape).prod()).reshape(shape).float()
    unconditional = conditional + 1000
    if batched:
        conditional = conditional.unsqueeze(0).repeat(2, 1, 1, 1, 1)
        unconditional = unconditional.unsqueeze(0)
        unconditional = unconditional.repeat(2, 1, 1, 1, 1)
    roi = torch.zeros(8, 2, 3, dtype=torch.bool)
    roi[:, 0, 1] = True
    result, count = zero_cfg_delta_outside_future_roi(
        torch=torch,
        conditional=conditional,
        unconditional=unconditional,
        future_roi=roi,
    )
    temporal_dim = result.ndim - 3
    assert torch.equal(result.select(temporal_dim, 0), unconditional.select(temporal_dim, 0))
    for frame in range(1, 9):
        actual = result.select(temporal_dim, frame)
        cond = conditional.select(temporal_dim, frame)
        uncond = unconditional.select(temporal_dim, frame)
        assert torch.equal(actual[..., 0, 1], uncond[..., 0, 1])
        background = torch.ones(2, 3, dtype=torch.bool)
        background[0, 1] = False
        assert torch.equal(actual[..., background], cond[..., background])
    expected_batch = int(conditional.shape[0]) if batched else 1
    expected_channels = int(conditional.shape[-4])
    assert count == expected_batch * expected_channels * 8 * 5


def test_zero_cfg_delta_validates_layout() -> None:
    with pytest.raises(ValueError, match="L0..L8"):
        zero_cfg_delta_outside_future_roi(
            torch=torch,
            conditional=torch.zeros(1, 8, 2, 2),
            unconditional=torch.zeros(1, 8, 2, 2),
            future_roi=torch.ones(8, 2, 2, dtype=torch.bool),
        )
