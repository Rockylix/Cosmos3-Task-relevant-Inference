# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Direct per-frame Core-96 with weighted action alignment and B0-start G1 profiling."""

from __future__ import annotations

from typing import Any

from cosmos_framework.scripts.robolab_v5_3_c_cond_dense_step0 import (
    V53CConditionalDenseStep0AllSparseLaterController,
)

STRATEGY_VERSION = "v7-direct-core96-weighted-g1-b0"
PROFILE_BLOCK_GROUPS = ((0, 11), (12, 19), (20, 27))
ACTION_HORIZON_WEIGHTS = (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0)
TOKEN_BUDGETS = (184, 152, 136)
STABLE_BUDGETS = (56, 40, 32)
CORE_TOKEN_BUDGET = 96


class V7DirectCore96WeightedG1B0Controller(V53CConditionalDenseStep0AllSparseLaterController):
    """Build one direct Core-96 mask per future frame and freeze it for the chunk."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs["profile_block_groups"] = PROFILE_BLOCK_GROUPS
        kwargs["action_horizon_weights"] = ACTION_HORIZON_WEIGHTS
        kwargs["token_budgets"] = TOKEN_BUDGETS
        kwargs["stable_budgets"] = STABLE_BUDGETS
        kwargs["core_token_budget"] = CORE_TOKEN_BUDGET
        # Equality makes stable_reference_core_masks the final direct Top-96 Core masks.
        kwargs["stable_reference_core_token_budget"] = CORE_TOKEN_BUDGET
        super().__init__(**kwargs)

    def finish(self) -> dict[str, Any]:
        summary = super().finish()
        if summary["core_token_budget"] != CORE_TOKEN_BUDGET:
            raise RuntimeError("V7 direct Core budget changed")
        if summary["stable_reference_core_token_budget"] != CORE_TOKEN_BUDGET:
            raise RuntimeError("V7 Core was not selected directly")
        summary.update(
            {
                "strategy_version": STRATEGY_VERSION,
                "experiment": "direct_core96_weighted_action_g1_b0",
                "core_selection": "per-frame direct Top-96 once per chunk",
                "core_fixed_across_block_groups_steps_and_cfg_branches": True,
                "profile_block_groups": [list(group) for group in PROFILE_BLOCK_GROUPS],
                "action_horizon_weights": list(ACTION_HORIZON_WEIGHTS),
            }
        )
        return summary
