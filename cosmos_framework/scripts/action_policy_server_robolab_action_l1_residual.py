# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboLab server for current-block Action-mass/L1-residual completion."""

from __future__ import annotations

import sys

from cosmos_framework.scripts import action_policy_server_robolab_action_mass90 as server
from cosmos_framework.scripts.robolab_action_attention_l1_residual_intervention import (
    ActionAttentionL1ResidualController,
)


def main() -> None:
    server.ActionAttentionMass90Controller = ActionAttentionL1ResidualController
    if "--intervention-output-dir" not in sys.argv:
        sys.argv.extend(
            [
                "--intervention-output-dir",
                "/root/robolab/cosmos-framework-edge-version1/experiments/preliminary/sparsity/action_attention_l1_residual/"
                "action_attention_current_block_mass90_l1_residual_shift5_BananaInBowlTask_sim_v1/server",
            ]
        )
    server.main()


if __name__ == "__main__":
    main()
