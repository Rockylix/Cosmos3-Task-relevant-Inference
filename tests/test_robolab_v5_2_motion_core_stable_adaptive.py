from __future__ import annotations

import torch

from cosmos_framework.scripts.robolab_v5_2_motion_core_stable_adaptive_velocity_cache import (
    action_aligned_future_raw_profiles,
    build_v52_ablation_plan,
)


def _records(spatial: int = 12) -> list[dict]:
    records = []
    for branch_scale, branch in ((1.0, "conditional"), (0.9, "unconditional")):
        for block in range(4, 10):
            profiles = torch.full((8, spatial), 0.01 * branch_scale)
            for frame in range(8):
                if block == 5:
                    profiles[frame].zero_()
                    profiles[frame, frame % spatial] = 1.0 * branch_scale
                else:
                    profiles[frame, (frame + block) % spatial] += (0.2 + 0.01 * block) * branch_scale
            records.append({"branch": branch, "block": block, "profiles": profiles})
    return records


def test_raw_profile_retains_true_future_frame_mass() -> None:
    spatial = 2
    num_gen = 9 * spatial + 33
    layout = {
        "num_gen_tokens": num_gen,
        "action_queries": [
            {"query_role": "predicted", "action_horizon": horizon, "gen_position": 9 * spatial + 1 + horizon}
            for horizon in range(32)
        ],
        "latent_positions": {
            f"L{latent}": list(range(latent * spatial, (latent + 1) * spatial)) for latent in range(9)
        },
    }
    q = torch.zeros(num_gen, 2, 4)
    k_ar = torch.zeros(3, 1, 4)
    k_gen = torch.zeros(num_gen, 1, 4)
    profiles = action_aligned_future_raw_profiles(
        torch=torch,
        q_gen=q,
        k_ar=k_ar,
        k_gen=k_gen,
        scaling=0.5,
        token_layout=layout,
    )
    assert profiles.shape == (8, spatial)
    expected_mass = spatial / (3 + num_gen)
    assert torch.allclose(profiles.sum(-1), torch.full((8,), expected_mass))
    assert not torch.allclose(profiles.sum(-1), torch.ones(8))


def test_v52_plan_selects_mass_and_concentration_core_and_nests_every_arm() -> None:
    plan = build_v52_ablation_plan(
        torch=torch,
        profile_records=_records(),
        block_groups=((4, 5), (6, 7), (8, 9)),
        token_budgets=(8, 6, 4),
        stable_budgets=(3, 2, 1),
        core_block_range=(4, 7),
        core_block_count=1,
        core_token_budget=1,
        max_replacements=1,
    )
    assert plan["core_blocks"] == [5]
    assert plan["core_masks"].shape == (8, 12)
    assert plan["stable_masks"].shape == (3, 12)
    assert not bool((plan["stable_masks"] & plan["core_masks"].any(0)).any())
    for mode in ("a", "b", "c", "d"):
        masks = plan["execution_masks"][mode]
        assert masks.sum(-1).tolist() == [[8] * 8, [6] * 8, [4] * 8]
        assert not bool((masks[2] & ~masks[1]).any())
        assert not bool((masks[1] & ~masks[0]).any())
    for mode in ("b", "c", "d"):
        masks = plan["execution_masks"][mode]
        assert torch.all(masks | ~plan["core_masks"].unsqueeze(0))
    assert len(plan["retention_rows"]) == 4 * 2 * 6 * 8
    assert all(0.0 <= row["attention_mass_retention"] <= 1.0 + 1e-6 for row in plan["retention_rows"])


def test_adaptive_zero_scores_are_budget_fill_not_adaptive() -> None:
    records = _records()
    for record in records:
        record["profiles"] = record["profiles"].mean(0, keepdim=True).expand(8, -1).clone()
    plan = build_v52_ablation_plan(
        torch=torch,
        profile_records=records,
        block_groups=((4, 5), (6, 7), (8, 9)),
        token_budgets=(8, 6, 4),
        stable_budgets=(3, 2, 1),
        core_block_range=(4, 7),
        core_block_count=1,
        core_token_budget=1,
        max_replacements=1,
    )
    assert torch.equal(plan["adaptive_scores"], torch.zeros_like(plan["adaptive_scores"]))
    labels = plan["category_labels"]["c"]
    assert not bool((labels == 3).any())
    assert bool((labels == 4).any())


def test_d_reports_forced_and_thresholded_replacements_separately() -> None:
    plan = build_v52_ablation_plan(
        torch=torch,
        profile_records=_records(),
        block_groups=((4, 5), (6, 7), (8, 9)),
        token_budgets=(8, 6, 4),
        stable_budgets=(3, 2, 1),
        core_block_range=(4, 7),
        core_block_count=1,
        core_token_budget=1,
        replacement_relative_threshold=0.05,
        max_replacements=1,
    )
    rows = plan["replacement_rows"]
    assert len(rows) == 3 * 8
    assert all(row["forced_replacements"] >= 0 for row in rows)
    assert all(0 <= row["threshold_accepted"] <= 1 for row in rows)
    assert all(row["threshold_rejected"] >= 0 for row in rows)
