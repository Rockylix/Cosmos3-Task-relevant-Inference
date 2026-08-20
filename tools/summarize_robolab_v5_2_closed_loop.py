# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Summarize the fixed-seed five-task V5.2 RoboLab closed-loop screening run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

ARMS = ("dense", "a", "b", "c", "d")
TASKS = (
    "BananaInBowlTask",
    "BananaOnPlateTask",
    "RubiksCubeTask",
    "RubiksCubeAndBananaTask",
    "RubiksCubeLeftOfBowlTask",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _p90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--robolab-output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=579362556)
    parser.add_argument("--dense-run-name", type=str, default=None)
    parser.add_argument("--sparse-run-suffix", type=str, default="json_v1")
    args = parser.parse_args()

    experiment_root = args.experiment_root.expanduser().resolve()
    robolab_root = args.robolab_output_root.expanduser().resolve()
    episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    by_arm: dict[str, dict[str, Any]] = {}

    for arm in ARMS:
        if arm == "dense":
            run_name = args.dense_run_name or f"v5_2_k80_dense_shift5_5tasks_seed{args.seed}_v1"
        else:
            run_name = f"v5_2_k80_{arm}_shift5_5tasks_seed{args.seed}_{args.sparse_run_suffix}"
        episode_path = robolab_root / run_name / "episode_results.jsonl"
        episodes = _read_jsonl(episode_path)
        if len(episodes) != len(TASKS):
            raise RuntimeError(f"{arm}: expected {len(TASKS)} episodes, found {len(episodes)}")
        if {item["task_name"] for item in episodes} != set(TASKS):
            raise RuntimeError(f"{arm}: task set does not match the fixed five-task suite")

        aggregate_path = experiment_root / arm / "server" / "aggregate.json"
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        request_times = [float(item["generation_wall_s"]) for item in aggregate["requests"]]
        warm_times = request_times[1:]
        if not warm_times:
            raise RuntimeError(f"{arm}: no warm requests found")

        token_summary = aggregate["requests"][0].get("token_savings")
        full_gen_tokens = int(token_summary["full_gen_tokens"]) if token_summary else None
        saved_all = float(token_summary["average_saved_gen_tokens_per_all_block_call"]) if token_summary else 0.0
        retained_ratio = 1.0 - saved_all / full_gen_tokens if full_gen_tokens else 1.0
        subset_violations = (
            int(token_summary["subset_violation_g3_not_g2"]) + int(token_summary["subset_violation_g2_not_g1"])
            if token_summary
            else 0
        )
        all_finite = bool(token_summary["all_finite"]) if token_summary else True

        successes = sum(bool(item["success"]) for item in episodes)
        summary = {
            "arm": arm,
            "successes": successes,
            "episodes": len(episodes),
            "success_rate": successes / len(episodes),
            "requests": len(request_times),
            "warm_generation_mean_s": statistics.mean(warm_times),
            "warm_generation_median_s": statistics.median(warm_times),
            "warm_generation_p90_s": _p90(warm_times),
            "mean_saved_gen_tokens_per_all_block_call": saved_all,
            "mean_gen_token_retained_ratio_all_block_calls": retained_ratio,
            "subset_violations": subset_violations,
            "all_finite": all_finite,
        }
        summary_rows.append(summary)
        by_arm[arm] = summary

        video_paths = {path.parent.name: str(path) for path in (robolab_root / run_name).glob("*/*_viewport.mp4")}
        for item in episodes:
            episode_rows.append(
                {
                    "arm": arm,
                    "task": item["task_name"],
                    "success": bool(item["success"]),
                    "episode_step": int(item["episode_step"]),
                    "score": float(item["score"]),
                    "reason": item.get("reason", ""),
                    "video": video_paths.get(item["task_name"], ""),
                }
            )

    dense_median = by_arm["dense"]["warm_generation_median_s"]
    for summary in summary_rows:
        summary["warm_generation_speedup_vs_dense"] = dense_median / summary["warm_generation_median_s"]

    _write_csv(
        experiment_root / "episode_results.csv",
        ["arm", "task", "success", "episode_step", "score", "reason", "video"],
        episode_rows,
    )
    _write_csv(
        experiment_root / "closed_loop_summary.csv",
        [
            "arm",
            "successes",
            "episodes",
            "success_rate",
            "requests",
            "warm_generation_mean_s",
            "warm_generation_median_s",
            "warm_generation_p90_s",
            "warm_generation_speedup_vs_dense",
            "mean_saved_gen_tokens_per_all_block_call",
            "mean_gen_token_retained_ratio_all_block_calls",
            "subset_violations",
            "all_finite",
        ],
        summary_rows,
    )

    payload = {
        "schema_version": 1,
        "experiment": "v5_2_motion_core_stable_adaptive_k80_closed_loop",
        "seed": args.seed,
        "prompt_format": "official_json",
        "shift": 5.0,
        "num_steps": 4,
        "tasks": list(TASKS),
        "episode_count_per_arm": len(TASKS),
        "arms": by_arm,
        "interpretation_boundary": (
            "One deterministic episode per task is a fixed-seed screening result, not a stable per-task success rate."
        ),
    }
    (experiment_root / "experiment.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    task_lookup = {(row["arm"], row["task"]): row for row in episode_rows}
    report = [
        "# V5.2 K80 五任务闭环筛选",
        "",
        f"固定 policy seed `{args.seed}`、RoboLab environment seed `0`、官方 JSON prompt、shift `5`、4-step；每个 arm 对每个任务只跑 1 个 episode。",
        "成功率按 `episode_results.jsonl` 的布尔字段 `success` 统计，不使用 `score`。",
        "",
        "| Arm | 成功数 | 成功率 | Warm chunk median (s) | P90 (s) | 相对 Dense | 全 block-call GEN token 保留率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary_rows:
        report.append(
            f"| {item['arm']} | {item['successes']}/{item['episodes']} | {100 * item['success_rate']:.1f}% | "
            f"{item['warm_generation_median_s']:.4f} | {item['warm_generation_p90_s']:.4f} | "
            f"{item['warm_generation_speedup_vs_dense']:.3f}x | "
            f"{100 * item['mean_gen_token_retained_ratio_all_block_calls']:.2f}% |"
        )
    report.extend(
        [
            "",
            "| Task | Dense | A | B | C | D |",
            "|---|:---:|:---:|:---:|:---:|:---:|",
        ]
    )
    for task in TASKS:
        marks = ["Y" if task_lookup[(arm, task)]["success"] else "N" for arm in ARMS]
        report.append(f"| {task} | {' | '.join(marks)} |")
    report.extend(
        [
            "",
            "## 结论边界",
            "",
            "- Dense 为 3/5；稀疏臂 A/B/C/D 分别为 3/5、2/5、4/5、3/5。当前固定 seed 筛选中 C 最高。",
            "- 相同 K80 预算下，C 恢复了 RubiksCube，而 A 与 Dense 的逐任务成败完全相同；B 丢失组合任务，D 丢失 BananaOnPlate。",
            "- 稀疏臂平均保留约 70.96% 的全 block-call GEN token，warm chunk median 约 0.727 s，对 Dense 约 1.20x。",
            "- C 的 4/5 只比 Dense 多一个固定 seed episode，不能解释为成功率显著提升；继续减 token 应同时保留 A 和 C 两条基线。",
            "- 每任务仅一个固定 seed episode，只适合作为继续减 token 前的筛选证据，不能作为稳定任务成功率。",
        ]
    )
    (experiment_root / "report_cn.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
