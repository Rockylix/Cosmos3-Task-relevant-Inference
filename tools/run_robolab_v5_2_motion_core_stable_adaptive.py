#!/usr/bin/env python3
"""Run paired Dense and V5.2 A-D ablations on one fixed RoboLab chunk."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from cosmos_framework.scripts.action_policy_server_robolab import RobolabServerArgs
from cosmos_framework.scripts.robolab_step0_fixed_roi_velocity_cache import GuidedVelocityTraceSampler
from cosmos_framework.scripts.robolab_v5_2_motion_core_stable_adaptive_velocity_cache import (
    DEFAULT_CORE_BLOCK_COUNT,
    DEFAULT_CORE_TOKEN_BUDGET,
    DEFAULT_K80_BUDGETS,
    DEFAULT_MAX_REPLACEMENTS,
    DEFAULT_REPLACEMENT_RELATIVE_THRESHOLD,
    DEFAULT_STABLE_BUDGETS,
    DEFAULT_STABLE_CV_PENALTY,
    V52MotionCoreStableAdaptiveController,
    V52VelocityCacheSampler,
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
    _metrics,
    _save_frames,
    _tensor,
    _uint8_frames,
)

OUTPUT_ROOT = Path(
    "/root/robolab/experiments/preliminary/sparsity/velocity_cache/"
    "v5_2_motion_core_stable_adaptive_k80_BananaInBowlTask_c3_v1"
)
ARM_NAMES = {
    "a": "a_v5_1",
    "b": "b_motion_core",
    "c": "c_stable_adaptive",
    "d": "d_v5_2_threshold",
}


def _paired_metrics(baseline: dict[str, Any], candidate: dict[str, Any], baseline_rgb: Any, candidate_rgb: Any) -> dict:
    baseline_action = _tensor(baseline["action"]).detach().cpu()[1:]
    candidate_action = _tensor(candidate["action"]).detach().cpu()[1:]
    baseline_vision = _tensor(baseline["vision"]).detach().cpu()
    candidate_vision = _tensor(candidate["vision"]).detach().cpu()
    return {
        "action": _metrics(baseline_action, candidate_action),
        "action_delta": _metrics(
            baseline_action[1:] - baseline_action[:-1], candidate_action[1:] - candidate_action[:-1]
        ),
        "action_jerk": _metrics(
            baseline_action[2:] - 2 * baseline_action[1:-1] + baseline_action[:-2],
            candidate_action[2:] - 2 * candidate_action[1:-1] + candidate_action[:-2],
        ),
        "vision_latent": _metrics(baseline_vision, candidate_vision),
        "decoded_rgb": _metrics(baseline_rgb, candidate_rgb),
    }


def _write_horizon_metrics(path: Path, baseline: Any, candidate: Any) -> None:
    baseline_action = _tensor(baseline["action"]).detach().cpu()[1:]
    candidate_action = _tensor(candidate["action"]).detach().cpu()[1:]
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["horizon", "mse", "relative_l2", "cosine", "max_absolute_error", "finite"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for horizon in range(int(baseline_action.shape[0])):
            writer.writerow({"horizon": horizon, **_metrics(baseline_action[horizon], candidate_action[horizon])})


def _report(metrics: dict[str, Any], args: argparse.Namespace, profile_max_abs_diff: float) -> str:
    d_summary = metrics["d"]["token_savings"]
    lines = [
        "# V5.2 Motion-Core / Stable-Adaptive K80 配对实验",
        "",
        f"- Task: `{args.task}`",
        f"- Chunk: `{args.chunk}`",
        f"- Seed: `{args.seed}`",
        f"- Shift: `{args.shift}`",
        f"- K80 budgets: `{list(args.group_token_budgets)}`",
        f"- A-D Step-0 raw profile 最大绝对差: `{profile_max_abs_diff:.3e}`",
        f"- Core blocks: `{d_summary['core_blocks']}`",
        "- Subset violations G3→G2 / G2→G1: "
        f"`{d_summary['subset_violation_g3_not_g2']} / {d_summary['subset_violation_g2_not_g1']}`",
        "",
        "| Arm | Action MSE | Action cos | Delta cos | Jerk cos | RGB cos | Latent cos | Wall (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, name in ARM_NAMES.items():
        item = metrics[mode]
        lines.append(
            f"| {name} | {item['action']['mse']:.6f} | {item['action']['cosine']:.6f} | "
            f"{item['action_delta']['cosine']:.6f} | {item['action_jerk']['cosine']:.6f} | "
            f"{item['decoded_rgb']['cosine']:.6f} | {item['vision_latent']['cosine']:.6f} | "
            f"{item['timing']['generation_wall_s']:.4f} |"
        )
    lines.extend(
        [
            "",
            "注意：这里是固定输入 chunk 的离线误差传播实验，不代表闭环任务成功率。",
            "单次 wall time 仅用于检查，不作为正式性能 benchmark。",
            "",
        ]
    )
    return "\n".join(lines)


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
    parser.add_argument("--group-token-budgets", type=int, nargs=3, default=DEFAULT_K80_BUDGETS)
    parser.add_argument("--stable-budgets", type=int, nargs=3, default=DEFAULT_STABLE_BUDGETS)
    parser.add_argument("--core-block-count", type=int, default=DEFAULT_CORE_BLOCK_COUNT)
    parser.add_argument("--core-token-budget", type=int, default=DEFAULT_CORE_TOKEN_BUDGET)
    parser.add_argument("--stable-cv-penalty", type=float, default=DEFAULT_STABLE_CV_PENALTY)
    parser.add_argument("--replacement-relative-threshold", type=float, default=DEFAULT_REPLACEMENT_RELATIVE_THRESHOLD)
    parser.add_argument("--max-replacements", type=int, default=DEFAULT_MAX_REPLACEMENTS)
    args = parser.parse_args()
    for name in ("output_root", "checkpoint", "conditioning_image", "episode_hdf5"):
        setattr(args, name, getattr(args, name).expanduser().absolute())
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

    dense_sampler = GuidedVelocityTraceSampler(service.model.sampler)
    print("[dense] paired reference", flush=True)
    with torch.inference_mode():
        dense, dense_wall_s = _generate(service, data_batch, args, dense_sampler)
    dense_rgb = _decode(service, dense["vision"])
    _save_frames(_uint8_frames(dense_rgb), args.output_root / "dense" / "future_frames")
    torch.save(dense, args.output_root / "dense" / "outputs.pt")

    metrics: dict[str, Any] = {}
    reference_profiles = None
    profile_max_abs_diff = 0.0
    output_rows = []
    for mode, name in ARM_NAMES.items():
        arm_root = args.output_root / name
        controller = V52MotionCoreStableAdaptiveController(
            ablation_mode=mode,
            torch=torch,
            net=service.model.net,
            guidance=args.guidance,
            num_steps=args.num_steps,
            threshold=args.threshold,
            token_budgets=args.group_token_budgets,
            stable_budgets=args.stable_budgets,
            core_block_count=args.core_block_count,
            core_token_budget=args.core_token_budget,
            stable_cv_penalty=args.stable_cv_penalty,
            replacement_relative_threshold=args.replacement_relative_threshold,
            max_replacements=args.max_replacements,
            output_dir=arm_root / "controller",
        )
        sampler = V52VelocityCacheSampler(
            service.model.sampler, controller, reference_velocities=dense_sampler.velocities
        )
        print(f"[{mode}] {name}", flush=True)
        with controller, torch.inference_mode():
            output, wall_s = _generate(service, data_batch, args, sampler)
        summary = controller.finish()
        sampler.save(arm_root / "controller")
        profiles = torch.stack(controller._profile_tensors)
        if reference_profiles is None:
            reference_profiles = profiles
        else:
            difference = float((profiles - reference_profiles).abs().max())
            profile_max_abs_diff = max(profile_max_abs_diff, difference)
            if difference > 1e-7:
                raise RuntimeError(f"A-D Step-0 profiles diverged: max_abs={difference}")

        rgb = _decode(service, output["vision"])
        _save_frames(_uint8_frames(rgb), arm_root / "future_frames")
        arm_metrics = _paired_metrics(dense, output, dense_rgb, rgb)
        arm_metrics["timing"] = {
            "dense_generation_wall_s": dense_wall_s,
            "generation_wall_s": wall_s,
            "single_run_speedup": dense_wall_s / wall_s,
            "single_run_only": True,
        }
        arm_metrics["token_savings"] = summary
        arm_metrics["all_finite"] = all(
            bool(arm_metrics[key]["finite"]) for key in ("action", "vision_latent", "decoded_rgb")
        )
        metrics[mode] = arm_metrics
        (arm_root / "metrics.json").write_text(
            json.dumps(arm_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        torch.save(output, arm_root / "outputs.pt")
        _write_horizon_metrics(arm_root / "action_horizon_metrics.csv", dense, output)
        output_rows.append(
            {
                "mode": mode,
                "name": name,
                "action_mse": arm_metrics["action"]["mse"],
                "action_cosine": arm_metrics["action"]["cosine"],
                "delta_cosine": arm_metrics["action_delta"]["cosine"],
                "jerk_cosine": arm_metrics["action_jerk"]["cosine"],
                "rgb_cosine": arm_metrics["decoded_rgb"]["cosine"],
                "rgb_relative_l2": arm_metrics["decoded_rgb"]["relative_l2"],
                "latent_cosine": arm_metrics["vision_latent"]["cosine"],
                "latent_relative_l2": arm_metrics["vision_latent"]["relative_l2"],
                "generation_wall_s": wall_s,
                "single_run_speedup": dense_wall_s / wall_s,
            }
        )

    metrics["dense_timing"] = {"generation_wall_s": dense_wall_s}
    metrics["profile_consistency"] = {
        "a_d_max_absolute_difference": profile_max_abs_diff,
        "tolerance": 1e-7,
        "passed": profile_max_abs_diff <= 1e-7,
    }
    (args.output_root / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_root / "output_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    config = {
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
        "group_token_budgets": list(args.group_token_budgets),
        "stable_budgets": list(args.stable_budgets),
        "core_block_count": args.core_block_count,
        "core_token_budget": args.core_token_budget,
        "stable_cv_penalty": args.stable_cv_penalty,
        "replacement_relative_threshold": args.replacement_relative_threshold,
        "max_replacements": args.max_replacements,
        "compile": False,
        "cuda_graphs": False,
        "paired_input_noise_rng": True,
    }
    (args.output_root / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_root / "report_cn.md").write_text(_report(metrics, args, profile_max_abs_diff), encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root), "metrics": output_rows}, ensure_ascii=False, indent=2))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
