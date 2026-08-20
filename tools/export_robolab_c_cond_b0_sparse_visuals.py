#!/usr/bin/env python3
"""Export decoded future frames and mask artifacts for the B0-sparse ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cosmos_framework.scripts.action_policy_server_robolab import RobolabServerArgs
from cosmos_framework.scripts.robolab_v5_2_motion_core_stable_adaptive_velocity_cache import (
    DEFAULT_CORE_BLOCK_COUNT,
    DEFAULT_CORE_TOKEN_BUDGET,
    DEFAULT_MAX_REPLACEMENTS,
    DEFAULT_REPLACEMENT_RELATIVE_THRESHOLD,
    DEFAULT_STABLE_BUDGETS,
    DEFAULT_STABLE_CV_PENALTY,
)
from cosmos_framework.scripts.robolab_v5_3_acd_packed_kernel_velocity_cache import V53VelocityCacheSampler
from cosmos_framework.scripts.robolab_v5_3_c_cond_dense_step0 import (
    V53CConditionalDenseStep0AllSparseLaterController,
)
from tools.run_robolab_step0_fixed_roi_velocity_cache import (
    CHECKPOINT,
    CONDITIONING_IMAGE,
    EPISODE_HDF5,
    PROMPT,
    SEED,
    _decode,
    _EagerRobolabPolicyService,
    _generate,
    _make_data_batch,
    _save_frames,
    _uint8_frames,
)

DEFAULT_OUTPUT = Path(
    "/root/robolab/cosmos-framework-edge-version1/experiments/preliminary/sparsity/visualization/"
    "all_sparse_later_future_overlay_BananaInBowlTask_c3_v1"
)


def _save_latent_aligned_frames(frames: np.ndarray, output_dir: Path) -> None:
    """Save decoded frames at latent indices L1..L8 (4 decoded frames per latent)."""

    output_dir.mkdir(parents=True)
    selected = []
    for latent in range(1, 9):
        decoded_index = latent * 4
        frame = np.ascontiguousarray(frames[decoded_index])
        Image.fromarray(frame).save(output_dir / f"L{latent}_frame_{decoded_index:03d}.png")
        selected.append(frame)
    Image.fromarray(np.concatenate(selected, axis=1)).save(output_dir / "latent_aligned_contact_sheet.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--conditioning-image", type=Path, default=CONDITIONING_IMAGE)
    parser.add_argument("--episode-hdf5", type=Path, default=EPISODE_HDF5)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--task", default="BananaInBowlTask")
    parser.add_argument("--chunk", type=int, default=3)
    parser.add_argument("--state-index", type=int, default=96)
    parser.add_argument("--finger-joint-index", type=int, default=7)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--group-token-budgets", type=int, nargs=3, default=(192, 160, 144))
    parser.add_argument("--stable-budgets", type=int, nargs=3, default=DEFAULT_STABLE_BUDGETS)
    args = parser.parse_args()
    if args.num_steps != 4 or float(args.shift) != 5.0:
        raise ValueError("This visualization is fixed to four denoise steps and shift=5")
    for name in ("output_root", "checkpoint", "conditioning_image", "episode_hdf5"):
        setattr(args, name, getattr(args, name).expanduser().absolute())
    if args.output_root.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.output_root}")
    controller_root = args.output_root / "controller"
    controller_root.mkdir(parents=True)

    service = _EagerRobolabPolicyService(
        RobolabServerArgs(
            checkpoint_path=str(args.checkpoint),
            format_prompt_as_json=True,
            seed=args.seed,
            deterministic_seed=True,
            guidance=args.guidance,
            num_steps=args.num_steps,
            shift=args.shift,
            guardrails=False,
        )
    )
    data_batch = _make_data_batch(service, args)
    controller = V53CConditionalDenseStep0AllSparseLaterController(
        ablation_mode="c",
        torch=torch,
        net=service.model.net,
        guidance=args.guidance,
        num_steps=args.num_steps,
        threshold=0.9,
        token_budgets=tuple(args.group_token_budgets),
        stable_budgets=tuple(args.stable_budgets),
        core_block_count=DEFAULT_CORE_BLOCK_COUNT,
        core_token_budget=DEFAULT_CORE_TOKEN_BUDGET,
        stable_cv_penalty=DEFAULT_STABLE_CV_PENALTY,
        replacement_relative_threshold=DEFAULT_REPLACEMENT_RELATIVE_THRESHOLD,
        max_replacements=DEFAULT_MAX_REPLACEMENTS,
        output_dir=controller_root,
        validate_intermediates=False,
        enable_nvtx=False,
    )
    sampler = V53VelocityCacheSampler(service.model.sampler, controller)
    with controller, torch.inference_mode():
        output, generation_wall_s = _generate(service, data_batch, args, sampler)
    summary = controller.finish()

    decoded = _uint8_frames(_decode(service, output["vision"]))
    _save_frames(decoded, args.output_root / "decoded_frames")
    _save_latent_aligned_frames(decoded, args.output_root / "latent_aligned_frames")
    torch.save(output, args.output_root / "outputs.pt")
    metadata = {
        "task": args.task,
        "chunk": args.chunk,
        "seed": args.seed,
        "guidance": args.guidance,
        "num_steps": args.num_steps,
        "shift": args.shift,
        "generation_wall_s_single_run": generation_wall_s,
        "decoded_frame_count_including_condition": int(decoded.shape[0]),
        "latent_aligned_decoded_indices": [latent * 4 for latent in range(1, 9)],
        "token_budgets": list(args.group_token_budgets),
        "all_finite": bool(summary["all_finite"]),
        "strategy_version": summary["strategy_version"],
    }
    (args.output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(args.output_root), **metadata}, ensure_ascii=False, indent=2))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
