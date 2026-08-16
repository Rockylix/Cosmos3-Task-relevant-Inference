from __future__ import annotations

import torch

from cosmos_framework.scripts.robolab_v5_2_motion_core_stable_adaptive_velocity_cache import (
    build_v52_ablation_plan,
)
from cosmos_framework.scripts.robolab_v6_core64_stable_fixed import (
    CORE_TOKEN_BUDGET,
    STABLE_BUDGETS,
    STABLE_REFERENCE_CORE_TOKEN_BUDGET,
    TOKEN_BUDGETS,
)


def _conditional_step0_records() -> list[dict]:
    spatial = 340
    base = torch.linspace(0.001, 0.340, spatial)
    records = []
    for block in range(4, 28):
        profiles = torch.stack([base + 1e-5 * block + 1e-7 * frame for frame in range(8)])
        records.append({"branch": "conditional", "block": block, "profiles": profiles})
    return records


def test_v6_selects_direct_core64_before_stable() -> None:
    assert STABLE_REFERENCE_CORE_TOKEN_BUDGET == CORE_TOKEN_BUDGET == 64

    plan = build_v52_ablation_plan(
        torch=torch,
        profile_records=_conditional_step0_records(),
        token_budgets=TOKEN_BUDGETS,
        stable_budgets=STABLE_BUDGETS,
        core_token_budget=CORE_TOKEN_BUDGET,
        stable_reference_core_token_budget=STABLE_REFERENCE_CORE_TOKEN_BUDGET,
    )

    # Equality proves that Core is selected directly as Top-64. There is no
    # Core-48 reference followed by a second expansion pass.
    assert torch.equal(plan["stable_reference_core_masks"], plan["core_masks"])
    assert plan["core_masks"].sum(-1).tolist() == [64] * 8

    # Stable is selected only after direct Core-64 and excludes the union of
    # Core positions from all eight future frames.
    core_union = plan["core_masks"].any(dim=0)
    assert not bool((plan["stable_masks"] & core_union).any())
    assert plan["stable_masks"].sum(-1).tolist() == list(STABLE_BUDGETS)

    masks = plan["execution_masks"]["c"]
    assert masks.sum(-1).tolist() == [[budget] * 8 for budget in TOKEN_BUDGETS]
    assert not bool((masks[2] & ~masks[1]).any())
    assert not bool((masks[1] & ~masks[0]).any())
