#!/usr/bin/env python3
"""Paired dense/oracle-sparse Cosmos3 Edge generation for one frozen RoboLab chunk."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
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
from cosmos_framework.scripts.robolab_action_attention_mass90_intervention import (
    ActionAttentionMass90Controller,
)

CHECKPOINT = Path("/root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID")
CONDITIONING_IMAGE = Path(
    "/root/robolab/cosmos-framework-edge/experiments/preliminary/representation/hidden_states/"
    "edge_hidden_banana_bowl_plate_c357_v1/"
    "task_pick_up_the_banana_and_place_it_in_the_bowl_dc79626d/"
    "chunk_000003/conditioning_observation.png"
)
EPISODE_HDF5 = Path(
    "/root/robolab/RoboLab/output/cosmos3_edge_hidden_banana_bowl_plate_c357_v1/BananaInBowlTask/run_0.hdf5"
)
OUTPUT_ROOT = Path(
    "/root/robolab/cosmos-framework-edge-version1/experiments/preliminary/sparsity/action_attention_mass90/"
    "action_attention_mass90_sharedmask_BananaInBowlTask_c3_shift1_v1"
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


def _metrics(reference: torch.Tensor, candidate: torch.Tensor, eps: float = 1e-12) -> dict[str, Any]:
    ref = reference.detach().float().reshape(-1).cpu()
    got = candidate.detach().float().reshape(-1).cpu()
    diff = got - ref
    return {
        "mse": float(diff.square().mean()),
        "relative_l2": float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(ref).clamp_min(eps)),
        "cosine": float(torch.nn.functional.cosine_similarity(ref, got, dim=0, eps=eps)),
        "max_absolute_error": float(diff.abs().max()),
        "finite": bool(torch.isfinite(got).all()),
    }


def _temporal_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.detach().float().cpu()
    got = candidate.detach().float().cpu()
    return {
        "action_delta": _metrics(ref[1:] - ref[:-1], got[1:] - got[:-1]),
        "action_jerk": _metrics(
            ref[2:] - 2 * ref[1:-1] + ref[:-2],
            got[2:] - 2 * got[1:-1] + got[:-2],
        ),
    }


def _uint8_frames(decoded: torch.Tensor) -> np.ndarray:
    video = decoded.detach().float().cpu()
    if video.ndim == 5 and video.shape[0] == 1:
        video = video[0]
    if video.ndim != 4 or int(video.shape[0]) != 3:
        raise RuntimeError(f"Expected decoded [C,T,H,W], got {tuple(video.shape)}")
    if float(video.min()) < 0.0:
        video = (video + 1.0) / 2.0
    return (video.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 3, 0).numpy()


def _save_frames(frames: np.ndarray, directory: Path) -> None:
    directory.mkdir(parents=True)
    for index, frame in enumerate(frames):
        Image.fromarray(frame).save(directory / f"frame_{index:03d}.png")


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
        "observation/gripper_position": np.asarray([[joint[args.finger_joint_index] / (np.pi / 4)]], np.float32),
        "prompt": args.prompt,
    }
    return _build_data_batch_from_sample(service._build_sample(observation))


def _generate(service: RobolabPolicyService, data_batch: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    _reset_rng(args.seed)
    torch.cuda.empty_cache()
    return service.model.generate_samples_from_batch(
        _clone(data_batch),
        guidance=args.guidance,
        seed=[args.seed],
        num_steps=args.num_steps,
        shift=args.shift,
    )


def _decode(service: RobolabPolicyService, vision: torch.Tensor) -> torch.Tensor:
    latent = vision[0] if vision.ndim == 5 and vision.shape[0] == 1 else vision
    with torch.inference_mode():
        return service.model.decode(latent)


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
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.9)
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

    print("[baseline] paired dense generation", flush=True)
    with torch.inference_mode():
        baseline = _generate(service, data_batch, args)

    print("[sparse] full probe + committed 90%-mass sparse generation", flush=True)
    controller = ActionAttentionMass90Controller(
        torch=torch,
        net=service.model.net,
        guidance=args.guidance,
        num_steps=args.num_steps,
        threshold=args.threshold,
        output_dir=args.output_root,
    )
    with controller, torch.inference_mode():
        sparse = _generate(service, data_batch, args)
    token_summary = controller.finish()

    baseline_action_with_condition = _tensor(baseline["action"]).detach().cpu()
    sparse_action_with_condition = _tensor(sparse["action"]).detach().cpu()
    # Row 0 is the conditioned current state and is trimmed by the real policy
    # server. Report only the 32 predicted actions actually returned to RoboLab.
    baseline_action = baseline_action_with_condition[1:]
    sparse_action = sparse_action_with_condition[1:]
    baseline_vision = _tensor(baseline["vision"]).detach().cpu()
    sparse_vision = _tensor(sparse["vision"]).detach().cpu()
    baseline_rgb = _decode(service, baseline_vision.to("cuda"))
    sparse_rgb = _decode(service, sparse_vision.to("cuda"))
    metrics = {
        "action": _metrics(baseline_action, sparse_action),
        **_temporal_metrics(baseline_action, sparse_action),
        "vision_latent": _metrics(baseline_vision, sparse_vision),
        "decoded_rgb": _metrics(baseline_rgb, sparse_rgb),
        "token_savings": token_summary,
    }
    (args.output_root / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "baseline_action_with_condition": baseline_action_with_condition,
            "sparse_action_with_condition": sparse_action_with_condition,
            "baseline_action": baseline_action,
            "sparse_action": sparse_action,
            "baseline_vision": baseline_vision,
            "sparse_vision": sparse_vision,
        },
        args.output_root / "paired_outputs.pt",
    )

    action_ref = baseline_action
    action_got = sparse_action
    with (args.output_root / "action_horizon_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["horizon", "mse", "relative_l2", "cosine", "max_absolute_error", "finite"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for horizon in range(int(action_ref.shape[0])):
            writer.writerow({"horizon": horizon, **_metrics(action_ref[horizon], action_got[horizon])})
    with (args.output_root / "action_joint_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["joint", "mse", "relative_l2", "cosine", "max_absolute_error", "finite"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for joint in range(int(action_ref.shape[1])):
            writer.writerow({"joint": joint, **_metrics(action_ref[:, joint], action_got[:, joint])})

    _save_frames(_uint8_frames(baseline_rgb), args.output_root / "baseline" / "future_frames")
    _save_frames(_uint8_frames(sparse_rgb), args.output_root / "sparse" / "future_frames")
    experiment = {
        "task": args.task,
        "chunk": args.chunk,
        "prompt": args.prompt,
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "guidance": args.guidance,
        "num_steps": args.num_steps,
        "shift": args.shift,
        "threshold": args.threshold,
        "compile": False,
        "cuda_graphs": False,
        "paired_input_noise_rng": True,
    }
    (args.output_root / "experiment.json").write_text(
        json.dumps(experiment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
