from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.scripts.robolab_grouped_temporal_closed_roi_velocity_cache import (
    VELOCITY_CACHE_STRATEGY_VERSION,
    GroupedTemporalClosedVelocityCacheSampler,
    build_grouped_score_smoothed_budgeted_masks,
    build_grouped_temporal_closed_masks,
    expand_grouped_token_roi_to_velocity_grid,
    grouped_roi_original_positions,
    merge_grouped_cached_background_velocity,
    nested_subset_local_positions,
)


def _layout(spatial: int = 4) -> dict:
    latents = {f"L{latent}": list(range(latent * spatial, (latent + 1) * spatial)) for latent in range(9)}
    action_start = 9 * spatial
    return {
        "num_gen_tokens": action_start + 33,
        "latent_positions": latents,
        "action_positions": list(range(action_start, action_start + 33)),
    }


def _records() -> list[dict]:
    records = []
    for branch in ("conditional", "unconditional"):
        for block in range(4, 10):
            group = (block - 4) // 2
            profiles = torch.zeros(8, 4)
            for frame in range(8):
                profiles[frame, (frame + group) % 4] = 1.0
            records.append({"branch": branch, "block": block, "profiles": profiles})
    return records


def test_strategy_version_is_v5_1() -> None:
    assert VELOCITY_CACHE_STRATEGY_VERSION == "v5.1"


def test_masks_use_exact_budgets_keep_frames_separate_and_nest_depth() -> None:
    normalized, smoothed, depth_scores, execution = build_grouped_temporal_closed_masks(
        torch=torch,
        profile_records=_records(),
        block_groups=((4, 5), (6, 7), (8, 9)),
        threshold=0.9,
        token_budgets=(3, 2, 1),
    )
    assert normalized.shape == smoothed.shape == depth_scores.shape == execution.shape == (3, 8, 4)
    assert torch.allclose(normalized.sum(-1), torch.ones(3, 8))
    assert execution.sum(-1).tolist() == [[3] * 8, [2] * 8, [1] * 8]
    assert torch.all(execution[0] | ~execution[1])
    assert torch.all(execution[1] | ~execution[2])


def test_score_smoothing_uses_available_neighbors_and_not_binary_union() -> None:
    scores = torch.zeros(1, 8, 4)
    scores[0, 0, 0] = 1.0
    scores[0, 1, 1] = 1.0
    scores[0, 2:, 2] = 1.0
    normalized, smoothed, depth_scores, execution = build_grouped_score_smoothed_budgeted_masks(
        torch=torch,
        scores=scores,
        token_budgets=(1,),
        temporal_weights=(0.25, 0.5, 0.25),
    )
    assert torch.allclose(normalized.sum(-1), torch.ones(1, 8))
    assert torch.allclose(smoothed[0, 0], torch.tensor([2 / 3, 1 / 3, 0.0, 0.0]))
    assert torch.allclose(smoothed[0, 1], torch.tensor([0.25, 0.5, 0.25, 0.0]))
    assert torch.equal(depth_scores, smoothed)
    assert execution.sum(-1).tolist() == [[1] * 8]
    assert torch.equal(execution[0, 0], torch.tensor([True, False, False, False]))
    assert torch.equal(execution[0, 1], torch.tensor([False, True, False, False]))


def test_budget_validation_rejects_deeper_mask_larger_than_shallow_mask() -> None:
    with pytest.raises(ValueError, match="non-increasing"):
        build_grouped_score_smoothed_budgeted_masks(
            torch=torch,
            scores=torch.ones(2, 8, 4),
            token_budgets=(2, 3),
        )


def test_grouped_positions_preserve_l0_action_and_frame_local_masks() -> None:
    layout = _layout()
    masks = torch.zeros(8, 4, dtype=torch.bool)
    for frame in range(8):
        masks[frame, frame % 4] = True
    selected = grouped_roi_original_positions(torch, layout, masks, torch.device("cpu"))
    selected_set = set(selected.tolist())
    assert set(layout["latent_positions"]["L0"]).issubset(selected_set)
    assert set(layout["action_positions"]).issubset(selected_set)
    for latent in range(1, 9):
        frame = layout["latent_positions"][f"L{latent}"]
        assert [position in selected_set for position in frame] == [spatial == (latent - 1) % 4 for spatial in range(4)]


def test_nested_subset_rejects_token_reentry() -> None:
    active = torch.tensor([0, 1, 3, 5, 8])
    target = torch.tensor([1, 5, 8])
    local = nested_subset_local_positions(torch, active, target)
    assert torch.equal(local, torch.tensor([1, 3, 4]))
    with pytest.raises(RuntimeError, match="re-enter"):
        nested_subset_local_positions(torch, active, torch.tensor([1, 2, 8]))


def test_grouped_velocity_merge_uses_per_frame_masks() -> None:
    current = torch.full((1, 9, 2, 2), 2.0)
    cached = torch.full_like(current, 1.0)
    masks = torch.zeros(8, 2, 2, dtype=torch.bool)
    for frame in range(8):
        masks[frame].reshape(-1)[frame % 4] = True
    merged, metrics = merge_grouped_cached_background_velocity(
        torch=torch,
        current=current,
        cached_step0=cached,
        frame_roi_mask=masks,
    )
    assert torch.equal(merged[:, 0], current[:, 0])
    for frame in range(8):
        selected = frame % 4
        flat = merged[:, frame + 1].reshape(1, -1)
        assert float(flat[0, selected]) == 2.0
        assert torch.equal(flat[0, torch.arange(4) != selected], torch.ones(3))
    assert metrics["background_elements"] == 24


def test_grouped_token_masks_expand_without_frame_mixing() -> None:
    masks = torch.zeros(8, 4, dtype=torch.bool)
    for frame in range(8):
        masks[frame, frame % 4] = True
    expanded = expand_grouped_token_roi_to_velocity_grid(
        torch=torch,
        frame_masks=masks,
        token_grid=(2, 2),
        velocity_grid=(4, 4),
    )
    assert expanded.shape == (8, 4, 4)
    assert int(expanded[0].sum()) == 4
    assert int(expanded[1].sum()) == 4
    assert not torch.equal(expanded[0], expanded[1])


def test_sampler_caches_step0_and_uses_final_frame_masks() -> None:
    class FakeInner:
        def __call__(self, velocity_fn, initial_noise, *, num_steps, **kwargs):
            del kwargs
            state = initial_noise
            for step in range(num_steps):
                state = velocity_fn(state, torch.tensor([[float(999 - step)]]))
            return state

    final_masks = torch.zeros(8, 4, dtype=torch.bool)
    final_masks[:, 0] = True
    controller = SimpleNamespace(
        torch=torch,
        vision_shape=(1, 9, 2, 2),
        execution_masks=torch.stack((torch.ones_like(final_masks), final_masks, final_masks)),
        num_steps=4,
        _layout={"latent_shape_thw": (9, 2, 2)},
    )
    call = 0

    def velocity_fn(noise_x, timestep):
        nonlocal call
        del noise_x, timestep
        call += 1
        return [torch.full((1 * 9 * 2 * 2 + 3,), float(call))]

    sampler = GroupedTemporalClosedVelocityCacheSampler(FakeInner(), controller)
    output = sampler(velocity_fn, [torch.zeros(39)], num_steps=4)
    vision = output[0][:-3].reshape(1, 9, 2, 2)
    assert torch.equal(vision[:, 0], torch.full_like(vision[:, 0], 4.0))
    assert torch.equal(vision[:, 1:, 0, 0], torch.full_like(vision[:, 1:, 0, 0], 4.0))
    background = vision[:, 1:].reshape(1, 8, 4)[..., 1:]
    assert torch.equal(background, torch.ones_like(background))
    assert torch.equal(output[0][-3:], torch.full((3,), 4.0))
    assert len(sampler.cache_trace) == 4
