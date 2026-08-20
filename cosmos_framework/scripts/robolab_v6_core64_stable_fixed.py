# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Version1: select direct Core-64 masks before constructing Stable masks.

``STRATEGY_VERSION`` keeps the historical V6-B identifier so existing
artifacts and analysis scripts remain readable. ``PUBLIC_VERSION`` is the
canonical name for new experiments.
"""

from __future__ import annotations

from typing import Any

from cosmos_framework.scripts.robolab_v5_3_c_cond_dense_step0 import (
    V53CConditionalDenseStep0AllSparseLaterController,
)

STRATEGY_VERSION = "v6-b-direct-core64-then-stable-k184-152-136"
PUBLIC_VERSION = "Version1"
TOKEN_BUDGETS = (184, 152, 136)
STABLE_BUDGETS = (88, 72, 64)
CORE_TOKEN_BUDGET = 64
STABLE_REFERENCE_CORE_TOKEN_BUDGET = CORE_TOKEN_BUDGET


class Version1Core64StableFixedController(V53CConditionalDenseStep0AllSparseLaterController):
    """Select direct Core-64 masks once, then build Stable and Adaptive tails."""

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
                "public_version": PUBLIC_VERSION,
                "former_public_name": "V6-B",
                "experiment": "direct_core64_then_stable_adaptive_reduced",
                "core_selection": "direct_top64_once_per_future_frame",
                "stable_selection": "after_direct_core64_excluding_cross_frame_core_union",
                "stable_forbidden_core_union_budget": STABLE_REFERENCE_CORE_TOKEN_BUDGET,
                "final_core_token_budget": CORE_TOKEN_BUDGET,
                "adaptive_and_fill_reduced_by_total_budget": True,
            }
        )
        return summary


# Backward-compatible import for historical tools and artifacts.
V6Core64StableFixedController = Version1Core64StableFixedController
