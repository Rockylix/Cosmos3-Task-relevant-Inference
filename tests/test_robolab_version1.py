from __future__ import annotations

import torch

from cosmos_framework.scripts import robolab_version1 as version1_module
from cosmos_framework.scripts.robolab_version1 import (
    ACTION_HORIZON_WEIGHTS,
    CORE_BLOCK_COUNT,
    CORE_TOKEN_BUDGET,
    STABLE_TOKEN_BUDGET,
    STRATEGY_VERSION,
    TOKEN_BUDGET,
    action_aligned_future_profiles,
    action_aligned_future_profiles_with_lse,
    build_core_stable_plan,
)


def _profile_records() -> list[dict]:
    generator = torch.Generator().manual_seed(73)
    return [
        {
            "block": block,
            "profiles": torch.rand((8, 340), generator=generator) + 0.01,
        }
        for block in range(28)
    ]


def test_core_then_stable_plan_is_exact_and_disjoint() -> None:
    plan = build_core_stable_plan(torch=torch, profile_records=_profile_records())

    assert tuple(plan["execution_mask"].shape) == (8, 340)
    assert plan["execution_mask"].sum(dim=-1).tolist() == [TOKEN_BUDGET] * 8
    assert plan["core_masks"].sum(dim=-1).tolist() == [CORE_TOKEN_BUDGET] * 8
    assert int(plan["stable_mask"].sum()) == STABLE_TOKEN_BUDGET
    assert int(plan["core_pool_mask"].sum()) == 340 - STABLE_TOKEN_BUDGET
    assert not bool((plan["core_masks"].any(dim=0) & plan["stable_mask"]).any())
    assert len(plan["core_blocks"]) == CORE_BLOCK_COUNT
    assert len(set(plan["core_blocks"])) == CORE_BLOCK_COUNT


def test_version1_constants_define_only_the_current_candidate() -> None:
    assert STRATEGY_VERSION == "version1-core64-stable120-k184-live-l0-no-velocity-cache"
    assert TOKEN_BUDGET == 184
    assert CORE_TOKEN_BUDGET == 64
    assert STABLE_TOKEN_BUDGET == 120
    assert ACTION_HORIZON_WEIGHTS == (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0)


def _profile_layout(spatial: int) -> dict:
    num_gen = 9 * spatial + 33
    return {
        "num_gen_tokens": num_gen,
        "action_queries": [
            {
                "query_role": "predicted",
                "action_horizon": horizon,
                "gen_position": 9 * spatial + 1 + horizon,
            }
            for horizon in range(32)
        ],
        "latent_positions": {
            f"L{latent}": list(range(latent * spatial, (latent + 1) * spatial)) for latent in range(9)
        },
    }


def test_action_relevance_uses_symmetric_center_weights() -> None:
    layout = _profile_layout(spatial=2)
    q = torch.zeros(layout["num_gen_tokens"], 2, 1)
    for horizon, value in enumerate((0.0, 1.0, 4.0, 10.0)):
        q[layout["action_queries"][horizon]["gen_position"], :, 0] = value
    k_ar = torch.zeros(3, 1, 1)
    k_gen = torch.zeros(layout["num_gen_tokens"], 1, 1)
    k_gen[layout["latent_positions"]["L1"][0], :, 0] = 1.0
    k_gen[layout["latent_positions"]["L1"][1], :, 0] = -1.0

    weighted = action_aligned_future_profiles(
        torch=torch,
        q_gen=q,
        k_ar=k_ar,
        k_gen=k_gen,
        scaling=1.0,
        token_layout=layout,
    )
    middle = action_aligned_future_profiles(
        torch=torch,
        q_gen=q,
        k_ar=k_ar,
        k_gen=k_gen,
        scaling=1.0,
        token_layout=layout,
        action_horizon_weights=(0, 0.5, 0.5, 0),
    )
    edges = action_aligned_future_profiles(
        torch=torch,
        q_gen=q,
        k_ar=k_ar,
        k_gen=k_gen,
        scaling=1.0,
        token_layout=layout,
        action_horizon_weights=(0.5, 0, 0, 0.5),
    )
    assert torch.allclose(weighted[0], (2.0 / 3.0) * middle[0] + (1.0 / 3.0) * edges[0])


def test_lse_profile_matches_full_softmax(monkeypatch) -> None:
    layout = _profile_layout(spatial=3)
    generator = torch.Generator().manual_seed(7)
    q = torch.randn(layout["num_gen_tokens"], 4, 8, generator=generator)
    k_ar = torch.randn(5, 2, 8, generator=generator)
    k_gen = torch.randn(layout["num_gen_tokens"], 2, 8, generator=generator)
    v_ar = torch.randn(5, 2, 8, generator=generator)
    v_gen = torch.randn(layout["num_gen_tokens"], 2, 8, generator=generator)

    def reference_attention(*, query, key, value, scale, return_lse, **kwargs):
        del kwargs
        repeat = query.shape[2] // key.shape[2]
        key = key.repeat_interleave(repeat, dim=2)
        value = value.repeat_interleave(repeat, dim=2)
        logits = torch.einsum("bqhd,bkhd->bhqk", query, key) * float(scale)
        probabilities = logits.softmax(dim=-1)
        output = torch.einsum("bhqk,bkhd->bqhd", probabilities, value)
        assert return_lse
        lse = logits.logsumexp(dim=-1).permute(0, 2, 1).unsqueeze(-1)
        return output, lse

    monkeypatch.setattr(version1_module, "attention", reference_attention)
    expected = action_aligned_future_profiles(
        torch=torch,
        q_gen=q,
        k_ar=k_ar,
        k_gen=k_gen,
        scaling=0.25,
        token_layout=layout,
    )
    actual = action_aligned_future_profiles_with_lse(
        torch=torch,
        q_gen=q,
        k_ar=k_ar,
        k_gen=k_gen,
        v_ar=v_ar,
        v_gen=v_gen,
        scaling=0.25,
        token_layout=layout,
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
