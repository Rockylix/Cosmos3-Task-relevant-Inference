"""Frozen, dependency-light definition of the approved ToCa scan."""

from itertools import product
from pathlib import Path

ROOT = Path("/root/robolab/worktrees/toca-future")
ROBOLAB = Path("/root/robolab/RoboLab")
EDGE_PYTHON = Path("/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python")
VAE = Path(
    "/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/"
    "921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
)
TASKS = [
    "BananaInBowlTask",
    "BananaOnPlateTask",
    "ButterAboveRaisinTask",
    "BowlStackingLeftOnRightTask",
    "GrabABagelTask",
    "LargerObjectRaisinBoxInBinTask",
    "MustardInLeftBinTask",
    "RubiksCubeTask",
    "RubiksCubeLeftOfBowlTask",
    "MarkerInMugTask",
]
DIFFICULTY = {task: "simple" if i < 8 else "moderate" for i, task in enumerate(TASKS)}


def configurations():
    result = {"dense": None}
    for schedule, bonus, ratio, cfg in product(
        ("dcdc", "dccc"), (0.0, 0.6), (0.1, 0.25, 0.5), ("shared", "independent")
    ):
        name = f"{schedule}_r{str(ratio).replace('.', 'p')}_b{int(bonus > 0)}_{cfg}"
        result[name] = {
            "full_steps": [0, 2] if schedule == "dcdc" else [0],
            "period": 2 if schedule == "dcdc" else 4,
            "fresh_ratio": ratio,
            "layer_slope": 0.5,
            "age_weight": 0.25,
            "attention_backend": "joint",
            "spatial_bonus": bonus,
            "cfg_selection": cfg,
        }
    return result
