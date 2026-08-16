# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""V6-A: expand Core while freezing the original Core-48 Stable mask."""

from __future__ import annotations

from typing import Any

from cosmos_framework.scripts.robolab_v5_3_c_cond_dense_step0 import (
    V53CConditionalDenseStep0AllSparseLaterController,
)

STRATEGY_VERSION = "v6-a-core64-stable-fixed-k184-152-136"
TOKEN_BUDGETS = (184, 152, 136)
STABLE_BUDGETS = (88, 72, 64)
CORE_TOKEN_BUDGET = 64
STABLE_REFERENCE_CORE_TOKEN_BUDGET = 48


class V6Core64StableFixedController(V53CConditionalDenseStep0AllSparseLaterController):
    """Use Core-64, the Core-48 Stable scaffold, and a smaller Adaptive tail."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs["token_budgets"] = TOKEN_BUDGETS
        kwargs["stable_budgets"] = STABLE_BUDGETS
        kwargs["core_token_budget"] = CORE_TOKEN_BUDGET
        kwargs["stable_reference_core_token_budget"] = STABLE_REFERENCE_CORE_TOKEN_BUDGET
        super().__init__(**kwargs)

    def finish(self) -> dict[str, Any]:
        summary = super().finish()
        summary.update(
            {
                "strategy_version": STRATEGY_VERSION,
                "experiment": "core64_stable_fixed_adaptive_reduced",
                "stable_mask_frozen_against_core_budget": STABLE_REFERENCE_CORE_TOKEN_BUDGET,
                "final_core_token_budget": CORE_TOKEN_BUDGET,
                "adaptive_and_fill_reduced_by_total_budget": True,
            }
        )
        return summary
