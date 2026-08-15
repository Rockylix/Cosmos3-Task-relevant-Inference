#!/usr/bin/env python3
"""Stable single-chunk benchmark for V5.2 reference and V5.3 optimized A/C/D."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from cosmos_framework.scripts.action_policy_server_robolab import RobolabServerArgs
from cosmos_framework.scripts.robolab_v5_2_motion_core_stable_adaptive_velocity_cache import (
    DEFAULT_CORE_BLOCK_COUNT,
    DEFAULT_CORE_TOKEN_BUDGET,
    DEFAULT_MAX_REPLACEMENTS,
    DEFAULT_REPLACEMENT_RELATIVE_THRESHOLD,
    DEFAULT_STABLE_BUDGETS,
    DEFAULT_STABLE_CV_PENALTY,
    V52MotionCoreStableAdaptiveController,
    V52VelocityCacheSampler,
)
from cosmos_framework.scripts.robolab_v5_3_acd_packed_kernel_velocity_cache import (
    V53OptimizedACDController,
    V53VelocityCacheSampler,
)
from tools.run_robolab_step0_fixed_roi_velocity_cache import (
    CHECKPOINT,
    CONDITIONING_IMAGE,
    EPISODE_HDF5,
    PROMPT,
    SEED,
    _clone,
    _EagerRobolabPolicyService,
    _make_data_batch,
    _metrics,
    _reset_rng,
    _tensor,
)

DEFAULT_OUTPUT = Path(
    "/root/robolab/experiments/preliminary/sparsity/velocity_cache/"
    "ac_budget_packed_kernel_shift5_BananaInBowlTask_c3_k80_v1"
)
MODES = ("dense", "a_ref", "a_opt", "c_ref", "c_opt", "d_ref", "d_opt")


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _controller_kwargs(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    return {
        "ablation_mode": mode,
        "torch": torch,
        "net": args.service.model.net,
        "guidance": args.guidance,
        "num_steps": args.num_steps,
        "threshold": 0.9,
        "token_budgets": tuple(args.group_token_budgets),
        "stable_budgets": tuple(args.stable_budgets),
        "core_block_count": args.core_block_count,
        "core_token_budget": args.core_token_budget,
        "stable_cv_penalty": args.stable_cv_penalty,
        "replacement_relative_threshold": args.replacement_relative_threshold,
        "max_replacements": args.max_replacements,
        "output_dir": None,
    }


def _run_once(args: argparse.Namespace, data_batch: dict[str, Any], label: str) -> tuple[dict[str, Any], float, Any]:
    controller = None
    sampler = args.service.model.sampler
    if label != "dense":
        mode, runtime = label.split("_", maxsplit=1)
        kwargs = _controller_kwargs(args, mode)
        if runtime == "ref":
            controller = V52MotionCoreStableAdaptiveController(**kwargs)
            sampler = V52VelocityCacheSampler(args.service.model.sampler, controller)
        elif runtime == "opt":
            controller = V53OptimizedACDController(
                **kwargs,
                validate_intermediates=False,
                enable_nvtx=args.enable_nvtx,
            )
            sampler = V53VelocityCacheSampler(args.service.model.sampler, controller)
        else:
            raise ValueError(f"Unknown runtime {runtime}")

    _reset_rng(args.seed)
    kwargs = {
        "sampler": sampler,
        "guidance": args.guidance,
        "seed": [args.seed],
        "num_steps": args.num_steps,
        "shift": args.shift,
    }
    torch.cuda.synchronize()
    start = time.perf_counter()
    if controller is None:
        output = args.service.model.generate_samples_from_batch(_clone(data_batch), **kwargs)
    else:
        with controller:
            output = args.service.model.generate_samples_from_batch(_clone(data_batch), **kwargs)
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - start
    summary = controller.finish() if controller is not None else None
    return output, wall_s, summary


def _output_metrics(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": _metrics(_tensor(reference["action"]), _tensor(candidate["action"])),
        "vision": _metrics(_tensor(reference["vision"]), _tensor(candidate["vision"])),
    }


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# A/C/D packed-kernel 稳定单 chunk benchmark",
        "",
        "主计时口径：同一已加载模型，compile/CUDA graph 关闭；每个 mode 预热后交替运行，",
        "CUDA 同步包围 `generate_samples_from_batch`，只报告稳定单 chunk median/P90。",
        "首请求、任务 wall time 和 RPC 时间不进入加速比。",
        "",
        "| Mode | Median (s) | P90 (s) | Mean (s) | Speedup vs Dense |",
        "|---|---:|---:|---:|---:|",
    ]
    dense = result["timing"]["dense"]["median_s"]
    for mode in result["modes"]:
        row = result["timing"][mode]
        lines.append(
            f"| {mode} | {row['median_s']:.6f} | {row['p90_s']:.6f} | "
            f"{row['mean_s']:.6f} | {dense / row['median_s']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "## 输出差异",
            "",
            "| Compare | Action max-abs | Action cosine | Vision max-abs | Vision cosine |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for comparison, item in result["output_metrics"].items():
        lines.append(
            f"| {comparison} | {item['action']['max_absolute_error']:.3e} | "
            f"{item['action']['cosine']:.9f} | {item['vision']['max_absolute_error']:.3e} | "
            f"{item['vision']['cosine']:.9f} |"
        )
    return "\n".join(lines) + "\n"


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
    parser.add_argument("--core-block-count", type=int, default=DEFAULT_CORE_BLOCK_COUNT)
    parser.add_argument("--core-token-budget", type=int, default=DEFAULT_CORE_TOKEN_BUDGET)
    parser.add_argument("--stable-cv-penalty", type=float, default=DEFAULT_STABLE_CV_PENALTY)
    parser.add_argument("--replacement-relative-threshold", type=float, default=DEFAULT_REPLACEMENT_RELATIVE_THRESHOLD)
    parser.add_argument("--max-replacements", type=int, default=DEFAULT_MAX_REPLACEMENTS)
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--measure-rounds", type=int, default=20)
    parser.add_argument("--schedule-seed", type=int, default=20260815)
    parser.add_argument("--enable-nvtx", action="store_true")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=MODES,
        help="Benchmark a subset; dense is required as the timing/output reference.",
    )
    args = parser.parse_args()
    if args.num_steps != 4 or args.shift != 5.0:
        raise ValueError("This experiment is fixed to four denoise steps and shift=5")
    modes = tuple(dict.fromkeys(args.modes))
    if "dense" not in modes:
        raise ValueError("--modes must include dense")
    for name in ("output_root", "checkpoint", "conditioning_image", "episode_hdf5"):
        setattr(args, name, getattr(args, name).expanduser().absolute())
    if args.output_root.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.output_root}")
    args.output_root.mkdir(parents=True)

    args.service = _EagerRobolabPolicyService(
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
    data_batch = _make_data_batch(args.service, args)

    # Unmeasured global warm-up initializes lazy CUDA/library state before any arm.
    _run_once(args, data_batch, "dense")
    rng = random.Random(args.schedule_seed)
    for round_index in range(args.warmup_rounds):
        schedule = list(modes)
        rng.shuffle(schedule)
        for mode in schedule:
            print(f"[warmup {round_index + 1}/{args.warmup_rounds}] {mode}", flush=True)
            _run_once(args, data_batch, mode)

    timings = {mode: [] for mode in modes}
    representative: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    schedule_rows = []
    for round_index in range(args.measure_rounds):
        schedule = list(modes)
        rng.shuffle(schedule)
        for order, mode in enumerate(schedule):
            print(f"[measure {round_index + 1}/{args.measure_rounds}] {mode}", flush=True)
            output, wall_s, summary = _run_once(args, data_batch, mode)
            timings[mode].append(wall_s)
            schedule_rows.append({"round": round_index, "order": order, "mode": mode, "wall_s": wall_s})
            if mode not in representative:
                representative[mode] = {
                    "action": _tensor(output["action"]).detach().cpu(),
                    "vision": _tensor(output["vision"]).detach().cpu(),
                }
                summaries[mode] = summary

    timing_summary = {
        mode: {
            "count": len(values),
            "mean_s": statistics.mean(values),
            "median_s": statistics.median(values),
            "p90_s": _percentile(values, 0.9),
            "min_s": min(values),
            "max_s": max(values),
            "samples_s": values,
        }
        for mode, values in timings.items()
    }
    output_metrics = {
        f"dense_vs_{mode}": _output_metrics(representative["dense"], representative[mode])
        for mode in modes
        if mode != "dense"
    }
    for arm in ("a", "c", "d"):
        if f"{arm}_ref" in representative and f"{arm}_opt" in representative:
            output_metrics[f"{arm}_ref_vs_{arm}_opt"] = _output_metrics(
                representative[f"{arm}_ref"], representative[f"{arm}_opt"]
            )
    result = {
        "schema_version": 1,
        "experiment": "acd_budget_group_packed_kernel_stable_single_chunk",
        "task": args.task,
        "chunk": args.chunk,
        "seed": args.seed,
        "guidance": args.guidance,
        "num_steps": args.num_steps,
        "shift": args.shift,
        "compile": False,
        "cuda_graphs": False,
        "timing_scope": "warm synchronized generate_samples_from_batch single chunk",
        "warmup_rounds_per_mode": args.warmup_rounds,
        "alternating_measure_rounds": args.measure_rounds,
        "modes": list(modes),
        "group_token_budgets": list(args.group_token_budgets),
        "stable_budgets": list(args.stable_budgets),
        "timing": timing_summary,
        "output_metrics": output_metrics,
        "representative_token_summaries": summaries,
    }
    (args.output_root / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_root / "timing_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", "order", "mode", "wall_s"])
        writer.writeheader()
        writer.writerows(schedule_rows)
    (args.output_root / "report_cn.md").write_text(_report(result), encoding="utf-8")
    print(_report(result), flush=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
