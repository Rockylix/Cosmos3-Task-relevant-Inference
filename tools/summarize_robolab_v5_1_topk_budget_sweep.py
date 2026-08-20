#!/usr/bin/env python3
"""Summarize a V5.1 fixed-budget RoboLab sweep from completed artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

CONFIGS = (100, 90, 80, 75, 70, 60)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--robolab-output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=579362556)
    args = parser.parse_args()

    episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for scale in CONFIGS:
        config = f"k{scale}"
        aggregate_path = args.experiment_root / config / "server" / "aggregate.json"
        result_dir = args.robolab_output_root / f"v5_1_topk_{config}_shift5_3tasks_seed{args.seed}_v1"
        result_path = result_dir / "episode_results.jsonl"
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        episodes = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line]
        if len(episodes) != 3:
            raise RuntimeError(f"Expected exactly three episodes for {config}, got {len(episodes)}")

        requests = aggregate["requests"]
        if not requests:
            raise RuntimeError(f"No completed policy requests for {config}")
        token_summary = requests[0]["token_savings"]
        budgets = token_summary["token_budgets"]
        group_rows = token_summary["block_group_token_savings"]
        warm_times = [float(request["generation_wall_s"]) for request in requests[1:]]
        if not warm_times:
            raise RuntimeError(f"No warm policy requests for {config}")

        for episode in episodes:
            task = str(episode["task_name"])
            video_matches = list((result_dir / task).glob("*_viewport.mp4"))
            episode_rows.append(
                {
                    "config": config,
                    "budget_scale_percent": scale,
                    "g1_tokens": budgets[0],
                    "g2_tokens": budgets[1],
                    "g3_tokens": budgets[2],
                    "task": task,
                    "success": bool(episode["success"]),
                    "episode_step": int(episode["episode_step"]),
                    "duration_s": float(episode["duration"]),
                    "policy_inference_s": float(episode["timing"]["policy_inference_s"]),
                    "policy_inference_avg_ms_per_sim_step": float(
                        episode["timing"]["policy_inference_avg_ms"]
                    ),
                    "wall_total_s": float(episode["timing"]["wall_total_s"]),
                    "reason": str(episode["reason"]),
                    "video": str(video_matches[0]) if len(video_matches) == 1 else "",
                }
            )

        success_count = sum(bool(episode["success"]) for episode in episodes)
        summary_rows.append(
            {
                "config": config,
                "budget_scale_percent": scale,
                "g1_tokens": budgets[0],
                "g2_tokens": budgets[1],
                "g3_tokens": budgets[2],
                "success_count": success_count,
                "task_count": len(episodes),
                "success_rate": success_count / len(episodes),
                "episode_steps_mean": statistics.fmean(int(episode["episode_step"]) for episode in episodes),
                "episode_steps_max": max(int(episode["episode_step"]) for episode in episodes),
                "completed_requests": int(aggregate["completed_requests"]),
                "cold_generation_wall_s": float(requests[0]["generation_wall_s"]),
                "warm_generation_mean_s": statistics.fmean(warm_times),
                "warm_generation_median_s": statistics.median(warm_times),
                "warm_generation_p90_s": _percentile(warm_times, 0.9),
                "g1_gen_retained_ratio": float(group_rows[0]["gen_retained_ratio"]),
                "g2_gen_retained_ratio": float(group_rows[1]["gen_retained_ratio"]),
                "g3_gen_retained_ratio": float(group_rows[2]["gen_retained_ratio"]),
                "average_saved_gen_tokens_per_sparse_block": float(
                    token_summary["average_saved_gen_tokens_per_sparse_block"]
                ),
                "all_finite": bool(token_summary["all_finite"]),
                "episode_results": str(result_path),
                "server_aggregate": str(aggregate_path),
            }
        )

    reference_warm = float(summary_rows[0]["warm_generation_median_s"])
    for row in summary_rows:
        row["warm_speedup_vs_k100"] = reference_warm / float(row["warm_generation_median_s"])
    if not all(bool(row["all_finite"]) for row in summary_rows):
        raise RuntimeError("At least one configuration reported NaN/Inf")
    _write_csv(args.experiment_root / "episode_results.csv", episode_rows)
    _write_csv(args.experiment_root / "budget_summary.csv", summary_rows)
    payload = {
        "schema_version": 1,
        "experiment": "v5_1_topk_budget_sweep_shift5_3tasks",
        "seed": args.seed,
        "tasks": ["BananaInBowlTask", "BananaOnPlateTask", "RubiksCubeTask"],
        "configurations": summary_rows,
        "all_finite": True,
        "scope": "one deterministic policy seed and one environment seed per task; not a stable success-rate estimate",
    }
    (args.experiment_root / "experiment.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
