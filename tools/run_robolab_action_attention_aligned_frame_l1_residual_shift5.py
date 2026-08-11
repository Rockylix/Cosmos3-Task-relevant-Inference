#!/usr/bin/env python3
"""Paired shift=5 aligned-frame 90%-mass/L1-residual experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import run_robolab_action_attention_mass90 as paired

from cosmos_framework.scripts.robolab_action_attention_aligned_frame_l1_residual_intervention import (
    ActionAttentionAlignedFrameL1ResidualController,
)

OUTPUT_ROOT = Path(
    "/root/robolab/experiments/preliminary/sparsity/action_attention_l1_residual/"
    "action_attention_aligned_frame_mass90_l1_residual_shift5_BananaInBowlTask_c3_v1"
)


def main() -> None:
    paired.ActionAttentionMass90Controller = ActionAttentionAlignedFrameL1ResidualController
    paired.OUTPUT_ROOT = OUTPUT_ROOT
    defaults = {"--shift": "5.0", "--threshold": "0.9"}
    for option, value in defaults.items():
        if option not in sys.argv:
            sys.argv.extend([option, value])
    paired.main()


if __name__ == "__main__":
    main()
