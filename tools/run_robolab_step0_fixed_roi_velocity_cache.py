#!/usr/bin/env python3
"""Paired dense vs step0-profile/fixed-ROI/background-velocity-cache run."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image

from cosmos_framework.scripts.action_policy_server_robolab import (
    RobolabPolicyService,
    RobolabServerArgs,
    _build_data_batch_from_sample,
)
from cosmos_framework.scripts.robolab_grouped_temporal_closed_roi_velocity_cache import (
    GroupedTemporalClosedROISparseController,
    GroupedTemporalClosedVelocityCacheSampler,
)
from cosmos_framework.scripts.robolab_step0_fixed_roi_velocity_cache import (
    GuidedBackgroundVelocityCacheSampler,
    GuidedVelocityTraceSampler,
    Step0FixedROISparseController,
)

CHECKPOINT = Path("/root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID")
CONDITIONING_IMAGE = Path(
    "/root/robolab/cosmos-framework-edge/experiments/preliminary/representation/hidden_states/"
    "edge_hidden_banana_bowl_plate_c357_v1/"
    "task_pick_up_the_banana_and_place_it_in_the_bowl_dc79626d/"
    "chunk_000003/conditioning_observation.png"
)
EPISODE_HDF5 = Path("/root/robolab/RoboLab/output/cosmos3_edge_banana/BananaInBowlTask/run_0.hdf5")
OUTPUT_ROOT = Path(
    "/root/robolab/cosmos-framework-edge-version1/experiments/preliminary/sparsity/velocity_cache/"
    "step0_profile_fixed_roi_velocity_cache_shift5_BananaInBowlTask_c3_v1"
)
PROMPT = "Pick up the banana and place it in the bowl"
SEED = 579362556


class _EagerRobolabPolicyService(RobolabPolicyService):
    def _build_setup_args(self, args: RobolabServerArgs) -> Any:
        setup = super()._build_setup_args(args)
        updates = {"use_torch_compile": False, "use_cuda_graphs": False}
        if "guardrails" in type(setup).model_fields:
            updates["guardrails"] = False
        if "offload_guardrail_models" in type(setup).model_fields:
            updates["offload_guardrail_models"] = False
        return setup.model_copy(update=updates)


def _reset_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return copy.deepcopy(value)


def _tensor(value: Any) -> torch.Tensor:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise RuntimeError(f"Expected one sample, got {len(value)}")
        value = value[0]
    if not torch.is_tensor(value):
        raise TypeError(f"Expected Tensor, got {type(value).__name__}")
    return value


def _metrics(reference: Any, candidate: Any, eps: float = 1e-12) -> dict[str, Any]:
    ref = _tensor(reference).detach().float().reshape(-1).cpu()
    got = _tensor(candidate).detach().float().reshape(-1).cpu()
    diff = got - ref
    return {
        "mse": float(diff.square().mean()),
        "relative_l2": float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(ref).clamp_min(eps)),
        "cosine": float(torch.nn.functional.cosine_similarity(ref, got, dim=0, eps=eps)),
        "max_absolute_error": float(diff.abs().max()),
        "finite": bool(torch.isfinite(got).all()),
    }


def _make_data_batch(service: RobolabPolicyService, args: argparse.Namespace) -> dict[str, Any]:
    image = np.asarray(Image.open(args.conditioning_image).convert("RGB"), dtype=np.uint8)
    with h5py.File(args.episode_hdf5, "r") as handle:
        joint = np.asarray(
            handle["data/demo_0/states/articulation/robot/joint_position"][args.state_index],
            dtype=np.float32,
        )
    observation = {
        "observation/image": image,
        "observation/joint_position": joint[:7][None, :],
        "observation/gripper_position": np.asarray([[joint[args.finger_joint_index] / (np.pi / 4)]], dtype=np.float32),
        "prompt": args.prompt,
    }
    return _build_data_batch_from_sample(service._build_sample(observation))


def _generate(
    service: RobolabPolicyService,
    data_batch: dict[str, Any],
    args: argparse.Namespace,
    sampler: Any,
) -> tuple[dict[str, Any], float]:
    _reset_rng(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = service.model.generate_samples_from_batch(
        _clone(data_batch),
        sampler=sampler,
        guidance=args.guidance,
        seed=[args.seed],
        num_steps=args.num_steps,
        shift=args.shift,
    )
    torch.cuda.synchronize()
    return output, time.perf_counter() - start


def _decode(service: RobolabPolicyService, vision: Any) -> torch.Tensor:
    latent = _tensor(vision).to("cuda")
    with torch.inference_mode():
        decoded = service.model.decode(latent)
    return decoded.detach().cpu()


def _uint8_frames(decoded: torch.Tensor) -> np.ndarray:
    video = decoded.float()
    if video.ndim == 5 and int(video.shape[0]) == 1:
        video = video[0]
    if video.ndim != 4 or int(video.shape[0]) != 3:
        raise RuntimeError(f"Expected decoded [C,T,H,W], got {tuple(video.shape)}")
    if float(video.min()) < 0.0:
        video = (video + 1.0) / 2.0
    return (video.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 3, 0).numpy()


def _save_frames(frames: np.ndarray, directory: Path) -> None:
    directory.mkdir(parents=True)
    for index, frame in enumerate(frames):
        Image.fromarray(np.ascontiguousarray(frame)).save(directory / f"frame_{index:03d}.png")
    Image.fromarray(np.concatenate(list(frames), axis=1)).save(directory.parent / "contact_sheet.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
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
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--first-sparse-block", type=int, default=4)
    parser.add_argument("--strategy-version", choices=("v1", "v5.1"), default="v1")
    parser.add_argument(
        "--group-token-budgets",
        type=int,
        nargs=3,
        metavar=("G1", "G2", "G3"),
        default=(240, 200, 180),
    )
    args = parser.parse_args()
    args.output_root = args.output_root.expanduser().absolute()
    args.checkpoint = args.checkpoint.expanduser().absolute()
    args.conditioning_image = args.conditioning_image.expanduser().absolute()
    args.episode_hdf5 = args.episode_hdf5.expanduser().absolute()
    if args.output_root.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.output_root}")
    args.output_root.mkdir(parents=True)

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

    baseline_sampler = GuidedVelocityTraceSampler(service.model.sampler)
    print("[baseline] dense paired generation", flush=True)
    with torch.inference_mode():
        baseline, baseline_wall_s = _generate(service, data_batch, args, baseline_sampler)

    if args.strategy_version == "v1":
        controller = Step0FixedROISparseController(
            torch=torch,
            net=service.model.net,
            guidance=args.guidance,
            num_steps=args.num_steps,
            threshold=args.threshold,
            first_sparse_block=args.first_sparse_block,
            output_dir=args.output_root / "controller",
        )
        sparse_sampler = GuidedBackgroundVelocityCacheSampler(
            service.model.sampler,
            controller,
            reference_velocities=baseline_sampler.velocities,
        )
    else:
        if args.first_sparse_block != 4:
            raise ValueError("V5.1 uses fixed block groups B4-B11/B12-B19/B20-B27")
        controller = GroupedTemporalClosedROISparseController(
            torch=torch,
            net=service.model.net,
            guidance=args.guidance,
            num_steps=args.num_steps,
            threshold=args.threshold,
            token_budgets=args.group_token_budgets,
            output_dir=args.output_root / "controller",
        )
        sparse_sampler = GroupedTemporalClosedVelocityCacheSampler(
            service.model.sampler,
            controller,
            reference_velocities=baseline_sampler.velocities,
        )
    print(f"[sparse] strategy={args.strategy_version} step0-profile ROI + velocity cache", flush=True)
    with controller, torch.inference_mode():
        sparse, sparse_wall_s = _generate(service, data_batch, args, sparse_sampler)
    controller_summary = controller.finish()
    sparse_sampler.save(args.output_root / "controller")

    baseline_action_full = _tensor(baseline["action"]).detach().cpu()
    sparse_action_full = _tensor(sparse["action"]).detach().cpu()
    baseline_action = baseline_action_full[1:]
    sparse_action = sparse_action_full[1:]
    baseline_vision = _tensor(baseline["vision"]).detach().cpu()
    sparse_vision = _tensor(sparse["vision"]).detach().cpu()
    baseline_rgb = _decode(service, baseline_vision)
    sparse_rgb = _decode(service, sparse_vision)
    metrics = {
        "action": _metrics(baseline_action, sparse_action),
        "action_delta": _metrics(baseline_action[1:] - baseline_action[:-1], sparse_action[1:] - sparse_action[:-1]),
        "action_jerk": _metrics(
            baseline_action[2:] - 2 * baseline_action[1:-1] + baseline_action[:-2],
            sparse_action[2:] - 2 * sparse_action[1:-1] + sparse_action[:-2],
        ),
        "vision_latent": _metrics(baseline_vision, sparse_vision),
        "decoded_rgb": _metrics(baseline_rgb, sparse_rgb),
        "timing": {
            "baseline_generation_wall_s": baseline_wall_s,
            "sparse_generation_wall_s": sparse_wall_s,
            "wall_speedup": baseline_wall_s / sparse_wall_s,
            "single_run_only": True,
        },
        "token_savings": controller_summary,
        "all_finite": all(
            bool(item["finite"])
            for item in (
                _metrics(baseline_action, sparse_action),
                _metrics(baseline_vision, sparse_vision),
                _metrics(baseline_rgb, sparse_rgb),
            )
        ),
    }
    (args.output_root / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "baseline_action_with_condition": baseline_action_full,
            "sparse_action_with_condition": sparse_action_full,
            "baseline_action": baseline_action,
            "sparse_action": sparse_action,
            "baseline_vision": baseline_vision,
            "sparse_vision": sparse_vision,
        },
        args.output_root / "paired_outputs.pt",
    )
    with (args.output_root / "action_horizon_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["horizon", "mse", "relative_l2", "cosine", "max_absolute_error", "finite"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for horizon in range(int(baseline_action.shape[0])):
            writer.writerow({"horizon": horizon, **_metrics(baseline_action[horizon], sparse_action[horizon])})

    _save_frames(_uint8_frames(baseline_rgb), args.output_root / "baseline" / "future_frames")
    _save_frames(_uint8_frames(sparse_rgb), args.output_root / "sparse" / "future_frames")
    (args.output_root / "experiment.json").write_text(
        json.dumps(
            {
                "task": args.task,
                "chunk": args.chunk,
                "prompt": args.prompt,
                "checkpoint": str(args.checkpoint),
                "conditioning_image": str(args.conditioning_image),
                "episode_hdf5": str(args.episode_hdf5),
                "state_index": args.state_index,
                "seed": args.seed,
                "guidance": args.guidance,
                "num_steps": args.num_steps,
                "shift": args.shift,
                "threshold": args.threshold,
                "first_sparse_block": args.first_sparse_block,
                "strategy_version": args.strategy_version,
                "group_token_budgets": list(args.group_token_budgets),
                "compile": False,
                "cuda_graphs": False,
                "paired_input_and_rng": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
