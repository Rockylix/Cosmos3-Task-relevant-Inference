from __future__ import annotations

from types import SimpleNamespace

import torch

from cosmos_framework.scripts.robolab_step0_fixed_roi_velocity_cache import (
    GuidedBackgroundVelocityCacheSampler,
    action_aligned_future_spatial_profiles,
    aggregate_fixed_roi,
    expand_token_roi_to_velocity_grid,
    fixed_roi_original_positions,
    merge_cached_background_velocity,
)


def _layout(spatial: int = 2) -> dict:
    latents = {
        f"L{latent}": list(range(latent * spatial, (latent + 1) * spatial)) for latent in range(9)
    }
    action_start = 9 * spatial
    return {
        "num_gen_tokens": action_start + 33,
        "latent_positions": latents,
        "action_positions": list(range(action_start, action_start + 33)),
        "action_queries": [
            {
                "gen_position": action_start + index,
                "query_role": "condition" if index == 0 else "predicted",
                "action_horizon": index - 1,
            }
            for index in range(33)
        ],
    }


def test_action_aligned_profiles_use_all_frames_and_normalize() -> None:
    layout = _layout()
    q_gen = torch.zeros(layout["num_gen_tokens"], 2, 4)
    k_ar = torch.zeros(3, 1, 4)
    k_gen = torch.zeros(layout["num_gen_tokens"], 1, 4)
    profiles = action_aligned_future_spatial_profiles(
        torch=torch,
        q_gen=q_gen,
        k_ar=k_ar,
        k_gen=k_gen,
        scaling=0.5,
        token_layout=layout,
    )
    assert profiles.shape == (8, 2)
    assert torch.allclose(profiles, torch.full((8, 2), 0.5))


def test_max_aggregate_mass_mask_is_fixed_and_nonempty() -> None:
    profiles = torch.tensor(
        [
            [[0.7, 0.2, 0.1, 0.0]] * 8,
            [[0.1, 0.7, 0.2, 0.0]] * 8,
        ],
        dtype=torch.float32,
    )
    mask, aggregate = aggregate_fixed_roi(torch, profiles, threshold=0.9)
    assert torch.equal(aggregate, torch.tensor([0.7, 0.7, 0.2, 0.0]))
    assert torch.equal(mask, torch.tensor([True, True, True, False]))


def test_fixed_roi_positions_preserve_l0_action_and_same_future_spatial_grid() -> None:
    layout = _layout(spatial=4)
    roi = torch.tensor([False, True, False, True])
    selected = fixed_roi_original_positions(torch, layout, roi, torch.device("cpu"))
    selected_set = set(selected.tolist())
    assert set(layout["latent_positions"]["L0"]).issubset(selected_set)
    assert set(layout["action_positions"]).issubset(selected_set)
    for latent in range(1, 9):
        frame = layout["latent_positions"][f"L{latent}"]
        assert [position in selected_set for position in frame] == [False, True, False, True]


def test_velocity_merge_uses_current_roi_and_step0_background() -> None:
    current = torch.full((2, 9, 2, 2), 2.0)
    cached = torch.full_like(current, 1.0)
    roi = torch.tensor([[True, False], [False, True]])
    merged, metrics = merge_cached_background_velocity(
        torch=torch,
        current=current,
        cached_step0=cached,
        roi_mask=roi,
    )
    assert torch.equal(merged[:, 0], current[:, 0])
    assert torch.equal(merged[:, 1:, roi], current[:, 1:, roi])
    assert torch.equal(merged[:, 1:, ~roi], cached[:, 1:, ~roi])
    assert metrics["background_elements"] == 2 * 8 * 2
    assert metrics["relative_l2"] == 0.5

    current_batched = current.unsqueeze(0)
    cached_batched = cached.unsqueeze(0)
    merged_batched, _ = merge_cached_background_velocity(
        torch=torch,
        current=current_batched,
        cached_step0=cached_batched,
        roi_mask=roi,
    )
    assert torch.equal(merged_batched[0], merged)


def test_token_roi_expands_to_odd_velocity_grid_without_reindexing() -> None:
    roi = torch.tensor([True, False, False, True])
    expanded = expand_token_roi_to_velocity_grid(
        torch=torch,
        roi_mask=roi,
        token_grid=(2, 2),
        velocity_grid=(3, 4),
    )
    assert torch.equal(
        expanded,
        torch.tensor(
            [
                [True, True, False, False],
                [True, True, False, False],
                [False, False, True, True],
            ]
        ),
    )


def test_sampler_caches_only_step0_and_merges_before_inner_update() -> None:
    class FakeInner:
        def __call__(self, velocity_fn, initial_noise, *, num_steps, **kwargs):
            del kwargs
            state = initial_noise
            for step in range(num_steps):
                state = velocity_fn(state, torch.tensor([[float(999 - step)]]))
            return state

    controller = SimpleNamespace(
        torch=torch,
        vision_shape=(1, 9, 2, 2),
        fixed_roi_mask=torch.tensor([True, False, False, True]),
        num_steps=4,
        _layout={"latent_shape_thw": (9, 2, 2)},
    )
    call = 0

    def velocity_fn(noise_x, timestep):
        nonlocal call
        del noise_x, timestep
        call += 1
        return [torch.full((1 * 9 * 2 * 2 + 3,), float(call))]

    sampler = GuidedBackgroundVelocityCacheSampler(FakeInner(), controller)
    output = sampler(velocity_fn, [torch.zeros(39)], num_steps=4)
    vision = output[0][:-3].reshape(1, 9, 2, 2)
    roi = controller.fixed_roi_mask.reshape(2, 2)
    assert torch.equal(vision[:, 1:, roi], torch.full_like(vision[:, 1:, roi], 4.0))
    assert torch.equal(vision[:, 1:, ~roi], torch.full_like(vision[:, 1:, ~roi], 1.0))
    assert torch.equal(output[0][-3:], torch.full((3,), 4.0))
    assert len(sampler.cache_trace) == 4
    assert sampler.cache_trace[0]["sample_0_relative_l2"] == 0.0
