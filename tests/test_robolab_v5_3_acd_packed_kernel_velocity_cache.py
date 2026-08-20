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
from cosmos_framework.scripts.robolab_v7_direct_core96_weighted_g1_b0 import (
    ACTION_HORIZON_WEIGHTS,
    CORE_TOKEN_BUDGET,
    PROFILE_BLOCK_GROUPS,
    STABLE_BUDGETS,
    TOKEN_BUDGETS,
)
from cosmos_framework.scripts.robolab_v8_core_stable_coverage_fill import (
    build_core_stable_coverage_fill_plan,
    compose_core_stable_fill_masks,
    coverage_strength_stable_masks,
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


def test_version1_branch_defaults_and_keeps_legacy_mode_alias() -> None:
    args = V53ServerArgs()
    assert args.ablation_mode == "version1"
    assert V53ServerArgs(ablation_mode="c_core64_stable_fixed_b0_sparse").ablation_mode == (
        "c_core64_stable_fixed_b0_sparse"
    )
    assert (args.roi_tokens_g1, args.roi_tokens_g2, args.roi_tokens_g3) == (192, 160, 144)
    assert args.stable_reference_core_token_budget is None
    assert V53ServerArgs(stable_reference_core_token_budget=48).stable_reference_core_token_budget == 48
    assert _percentile([], 0.9) is None
    assert _percentile([1.0, 2.0, 3.0], 0.9) == pytest.approx(2.8)


def test_v7_direct_core_weighted_profile_constants() -> None:
    args = V53ServerArgs(ablation_mode="c_direct_core96_weighted_g1_b0")
    assert args.ablation_mode == "c_direct_core96_weighted_g1_b0"
    assert PROFILE_BLOCK_GROUPS == ((0, 11), (12, 19), (20, 27))
    assert ACTION_HORIZON_WEIGHTS == pytest.approx((1 / 6, 1 / 3, 1 / 3, 1 / 6))
    assert CORE_TOKEN_BUDGET == 96
    assert STABLE_BUDGETS == (56, 40, 32)
    assert TOKEN_BUDGETS == (184, 152, 136)


def test_server_accepts_core96_coverage_stable_fill_mode() -> None:
    args = V53ServerArgs(ablation_mode="c_core96_coverage_stable_fill")
    assert args.ablation_mode == "c_core96_coverage_stable_fill"


def test_coverage_strength_uses_six_of_eight_and_third_smallest() -> None:
    scores = torch.zeros(3, 8, 12)
    for group in range(3):
        for frame in range(8):
            scores[group, frame, 8] = 0.1
            if frame < 6:
                scores[group, frame, 0] = 1.0 + group
                scores[group, frame, 1] = 2.0 + group
                scores[group, frame, 2] = 3.0 + group
                scores[group, frame, 3] = 0.5
            else:
                scores[group, frame, 9:12] = torch.tensor([4.0, 5.0, 6.0])
    result = coverage_strength_stable_masks(
        torch=torch,
        group_raw_scores=scores,
        stable_budgets=(3, 2, 1),
        top_r=4,
        min_frame_count=6,
        strength_order=3,
    )
    assert result["coverage_counts"][:, :3].tolist() == [[6, 6, 6]] * 3
    assert result["stable_masks"].sum(-1).tolist() == [3, 2, 1]
    assert bool(result["stable_masks"][2, 2])
    assert not bool((result["stable_masks"][2] & ~result["stable_masks"][1]).any())
    assert not bool((result["stable_masks"][1] & ~result["stable_masks"][0]).any())


def test_core_stable_overlap_is_allowed_and_fill_keeps_exact_nested_budgets() -> None:
    core = torch.zeros(8, 12, dtype=torch.bool)
    core[:, :4] = True
    stable = torch.zeros(3, 12, dtype=torch.bool)
    stable[2, :2] = True
    stable[1, :3] = True
    stable[0, :4] = True
    scores = torch.arange(3 * 8 * 12, dtype=torch.float32).reshape(3, 8, 12)
    result = compose_core_stable_fill_masks(
        torch=torch,
        core_masks=core,
        stable_masks=stable,
        group_raw_scores=scores,
        token_budgets=(8, 7, 6),
    )
    masks = result["masks"]
    assert masks.sum(-1).tolist() == [[8] * 8, [7] * 8, [6] * 8]
    assert result["core_stable_overlap_counts"].tolist() == [[4] * 8, [3] * 8, [2] * 8]
    assert not bool((masks[2] & ~masks[1]).any())
    assert not bool((masks[1] & ~masks[0]).any())
    assert not bool((result["labels"] == 3).any())


def test_core_stable_coverage_fill_plan_is_exact_nested_and_has_no_adaptive() -> None:
    records = []
    for block in range(6):
        profiles = torch.full((8, 24), 0.01)
        for frame in range(8):
            profiles[frame, frame % 6] += 1.0
            profiles[frame, 6 + block] += 0.5
        records.append({"branch": "conditional", "block": block, "profiles": profiles})
    plan = build_core_stable_coverage_fill_plan(
        torch=torch,
        profile_records=records,
        block_groups=((0, 1), (2, 3), (4, 5)),
        token_budgets=(14, 12, 10),
        stable_budgets=(5, 3, 2),
        core_block_range=(1, 4),
        core_block_count=2,
        core_token_budget=4,
        top_r=8,
        min_frame_count=6,
        strength_order=3,
    )
    masks = plan["execution_masks"]["c"]
    assert masks.sum(-1).tolist() == [[14] * 8, [12] * 8, [10] * 8]
    assert not bool((masks[2] & ~masks[1]).any())
    assert not bool((masks[1] & ~masks[0]).any())
    assert torch.equal(plan["adaptive_scores"], torch.zeros_like(plan["adaptive_scores"]))
    assert not bool((plan["category_labels"]["c"] == 3).any())


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
