from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.scripts.action_policy_server_robolab_v5_3_acd_packed_kernel import (
    V53ServerArgs,
    _percentile,
)
from cosmos_framework.scripts.robolab_grouped_temporal_closed_roi_velocity_cache import (
    GroupedTemporalClosedVelocityCacheSampler,
)
from cosmos_framework.scripts.robolab_v5_3_acd_packed_kernel_velocity_cache import (
    V53VelocityCacheSampler,
    subset_local_positions_searchsorted,
)


def test_searchsorted_subset_returns_local_positions_and_rejects_reentry() -> None:
    active = torch.tensor([0, 1, 3, 5, 8])
    target = torch.tensor([1, 5, 8])
    assert torch.equal(
        subset_local_positions_searchsorted(torch, active, target),
        torch.tensor([1, 3, 4]),
    )
    with pytest.raises(RuntimeError, match="not a sorted subset"):
        subset_local_positions_searchsorted(torch, active, torch.tensor([1, 2, 8]))


def test_searchsorted_subset_handles_empty_target() -> None:
    active = torch.tensor([0, 2, 4])
    target = torch.empty(0, dtype=torch.long)
    result = subset_local_positions_searchsorted(torch, active, target)
    assert result.dtype == torch.long
    assert result.numel() == 0


def test_v53_closed_loop_defaults_and_percentile() -> None:
    args = V53ServerArgs()
    assert args.ablation_mode == "c"
    assert (args.roi_tokens_g1, args.roi_tokens_g2, args.roi_tokens_g3) == (192, 160, 144)
    assert _percentile([], 0.9) is None
    assert _percentile([1.0, 2.0, 3.0], 0.9) == pytest.approx(2.8)


def test_v53_fast_velocity_sampler_matches_instrumented_v52_sampler() -> None:
    class FakeInner:
        def __call__(self, velocity_fn, initial_noise, *, num_steps, **kwargs):
            del kwargs
            state = initial_noise
            for step in range(num_steps):
                state = velocity_fn(state, torch.tensor([[float(999 - step)]]))
            return state

    final_masks = torch.zeros(8, 4, dtype=torch.bool)
    for frame in range(8):
        final_masks[frame, frame % 4] = True
    controller = SimpleNamespace(
        torch=torch,
        vision_shape=(1, 9, 2, 2),
        execution_masks=torch.stack((torch.ones_like(final_masks), final_masks, final_masks)),
        num_steps=4,
        _layout={"latent_shape_thw": (9, 2, 2)},
    )

    def make_velocity_fn():
        call = 0

        def velocity_fn(noise_x, timestep):
            nonlocal call
            del noise_x, timestep
            call += 1
            values = torch.arange(39, dtype=torch.float32) + 100.0 * call
            return [values]

        return velocity_fn

    initial = [torch.zeros(39)]
    reference = GroupedTemporalClosedVelocityCacheSampler(FakeInner(), controller)(
        make_velocity_fn(),
        initial,
        num_steps=4,
    )
    optimized = V53VelocityCacheSampler(FakeInner(), controller)(
        make_velocity_fn(),
        initial,
        num_steps=4,
    )
    assert len(reference) == len(optimized) == 1
    assert torch.equal(reference[0], optimized[0])
