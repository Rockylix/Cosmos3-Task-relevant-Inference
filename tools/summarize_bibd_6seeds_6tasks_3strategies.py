#!/usr/bin/env python3
"""Summarize the 6-seed BIBD closed-loop comparison and stable chunk timing."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from tools.run_bibd_6seeds_6tasks_3strategies import (
    EXPERIMENT_ROOT,
    ROBOLAB_ROOT,
    SEED_PAIRS,
    TASKS,
    tasks_for_pair,
)

STRATEGIES = ("baseline", "version1", "direct_core64")
DISPLAY_NAMES = {
    "baseline": "Dense baseline",
    "version1": "Version1",
    "direct_core64": "Direct Core64 + Stable (V6-B)",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return center - radius, center + radius


def basic_stats(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = list(values)
    return {
        "count": len(samples),
        "mean": statistics.mean(samples) if samples else None,
        "median": statistics.median(samples) if samples else None,
        "p90": percentile(samples, 0.9),
        "min": min(samples) if samples else None,
        "max": max(samples) if samples else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_episode_rows() -> list[dict[str, Any]]:
    difficulty_by_task = {
        task: difficulty for difficulty, task_map in TASKS.items() for task in task_map.values()
    }
    letter_by_difficulty_task = {
        (difficulty, task): letter
        for difficulty, task_map in TASKS.items()
        for letter, task in task_map.items()
    }
    rows: list[dict[str, Any]] = []
    for seed, pair in SEED_PAIRS:
        expected = tasks_for_pair(pair)
        for strategy in STRATEGIES:
            result_path = (
                ROBOLAB_ROOT
                / "output"
                / f"bibd6x6_{strategy}_seed{seed}_v1"
                / "episode_results.jsonl"
            )
            records = read_jsonl(result_path)
            by_task = {str(record["task_name"]): record for record in records if int(record.get("run", 0)) == 0}
            missing = sorted(set(expected) - set(by_task))
            unexpected = sorted(set(by_task) - set(expected))
            if missing or unexpected or len(by_task) != 6:
                raise RuntimeError(
                    f"Incomplete result {strategy=} {seed=}: missing={missing}, unexpected={unexpected}, "
                    f"rows={len(by_task)} path={result_path}"
                )
            for task in expected:
                item = by_task[task]
                difficulty = difficulty_by_task[task]
                timing = item.get("timing", {})
                rows.append(
                    {
                        "strategy": strategy,
                        "strategy_display": DISPLAY_NAMES[strategy],
                        "policy_seed": seed,
                        "pair": pair,
                        "difficulty": difficulty,
                        "task_letter": letter_by_difficulty_task[(difficulty, task)],
                        "task": task,
                        "success": bool(item["success"]),
                        "episode_step": int(item["episode_step"]),
                        "official_max_steps": int(item.get("official_max_steps", item["episode_step"])),
                        "effective_max_steps": int(item.get("effective_max_steps", item["episode_step"])),
                        "terminated_by_eval_cap": bool(item.get("terminated_by_eval_cap", False)),
                        "policy_inference_s": timing.get("policy_inference_s"),
                        "policy_inference_avg_ms": timing.get("policy_inference_avg_ms"),
                        "wall_total_s": timing.get("wall_total_s"),
                        "result_path": str(result_path),
                    }
                )
    if len(rows) != len(SEED_PAIRS) * 6 * len(STRATEGIES):
        raise RuntimeError(f"Expected 108 episode rows, got {len(rows)}")
    return rows


def success_summary(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for group_key, group in sorted(groups.items()):
        successes = sum(bool(row["success"]) for row in group)
        low, high = wilson(successes, len(group))
        success_steps = [float(row["episode_step"]) for row in group if row["success"]]
        all_steps = [float(row["episode_step"]) for row in group]
        row = {key: value for key, value in zip(keys, group_key, strict=True)}
        row.update(
            {
                "successes": successes,
                "episodes": len(group),
                "success_rate": successes / len(group),
                "success_rate_ci95_low": low,
                "success_rate_ci95_high": high,
                "success_steps_mean": statistics.mean(success_steps) if success_steps else None,
                "success_steps_median": statistics.median(success_steps) if success_steps else None,
                "success_steps_p90": percentile(success_steps, 0.9),
                "all_steps_mean": statistics.mean(all_steps),
                "all_steps_median": statistics.median(all_steps),
                "eval_cap_failures": sum(bool(row["terminated_by_eval_cap"]) for row in group),
            }
        )
        output.append(row)
    return output


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        cells[(int(row["policy_seed"]), str(row["task"]))][str(row["strategy"])] = row
    output: list[dict[str, Any]] = []
    for (seed, task), arms in sorted(cells.items()):
        if set(arms) != set(STRATEGIES):
            raise RuntimeError(f"Unpaired cell seed={seed} task={task}: {sorted(arms)}")
        base = arms["baseline"]
        item: dict[str, Any] = {
            "policy_seed": seed,
            "pair": base["pair"],
            "difficulty": base["difficulty"],
            "task": task,
        }
        for strategy in STRATEGIES:
            item[f"{strategy}_success"] = arms[strategy]["success"]
            item[f"{strategy}_episode_step"] = arms[strategy]["episode_step"]
        for candidate in ("version1", "direct_core64"):
            item[f"{candidate}_success_minus_baseline"] = int(arms[candidate]["success"]) - int(base["success"])
            item[f"{candidate}_steps_minus_baseline"] = (
                int(arms[candidate]["episode_step"]) - int(base["episode_step"])
                if arms[candidate]["success"] and base["success"]
                else None
            )
        output.append(item)
    if len(output) != len(SEED_PAIRS) * 6:
        raise RuntimeError(f"Expected 36 paired cells, got {len(output)}")
    return output


def load_closed_loop_timings(experiment_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        for seed_dir in sorted((experiment_root / "closed_loop" / strategy).glob("seed_*")):
            seed = int(seed_dir.name.split("_", 1)[1])
            for aggregate_path in sorted(seed_dir.glob("server*/aggregate.json")):
                aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
                requests = aggregate.get("requests", [])
                for index, request in enumerate(requests):
                    samples.append(
                        {
                            "strategy": strategy,
                            "policy_seed": seed,
                            "server_run": aggregate_path.parent.name,
                            "request_index": int(request.get("request_index", index)),
                            "is_server_warm": index > 0,
                            "generation_wall_s": float(request["generation_wall_s"]),
                            "aggregate_path": str(aggregate_path),
                        }
                    )
    summary: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        warm = [row["generation_wall_s"] for row in samples if row["strategy"] == strategy and row["is_server_warm"]]
        stats = basic_stats(warm)
        summary.append({"strategy": strategy, **{f"warm_{key}": value for key, value in stats.items()}})
    return samples, summary


def fmt(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def exact_mcnemar_p(candidate_only: int, baseline_only: int) -> float:
    discordant = candidate_only + baseline_only
    if discordant == 0:
        return 1.0
    tail = min(candidate_only, baseline_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def build_report(
    overall: list[dict[str, Any]],
    difficulty: list[dict[str, Any]],
    by_seed: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    closed_timing: list[dict[str, Any]],
    stable_metrics: dict[str, Any] | None,
    visual_root: Path | None,
) -> str:
    overall_by = {row["strategy"]: row for row in overall}
    timing_by = {row["strategy"]: row for row in closed_timing}
    lines = [
        "# Dense / Version1 / Direct Core64+Stable 多 seed 闭环对比",
        "",
        "## 闭环主结果",
        "",
        "成功严格取自 `episode_results.jsonl.success`。完成步数只在成功 episode 内聚合；失败或达到上限的轨迹不混入成功步数。",
        "",
        "| Strategy | Success | Rate | 95% Wilson CI | Success steps mean | median | P90 | Cap failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in STRATEGIES:
        row = overall_by[strategy]
        lines.append(
            f"| {DISPLAY_NAMES[strategy]} | {row['successes']}/{row['episodes']} | {100*row['success_rate']:.1f}% | "
            f"[{100*row['success_rate_ci95_low']:.1f}, {100*row['success_rate_ci95_high']:.1f}] | "
            f"{fmt(row['success_steps_mean'])} | {fmt(row['success_steps_median'])} | "
            f"{fmt(row['success_steps_p90'])} | {row['eval_cap_failures']} |"
        )
    lines.extend(
        [
            "",
            "## 按难度",
            "",
            "| Difficulty | Strategy | Success | Rate | Success steps median |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in difficulty:
        lines.append(
            f"| {row['difficulty']} | {DISPLAY_NAMES[row['strategy']]} | {row['successes']}/{row['episodes']} | "
            f"{100*row['success_rate']:.1f}% | {fmt(row['success_steps_median'])} |"
        )
    seed_lookup = {(int(row["policy_seed"]), str(row["strategy"])): row for row in by_seed}
    lines.extend(
        [
            "",
            "## 每个 policy seed 的成功数",
            "",
            "| Strategy | " + " | ".join(str(seed) for seed, _ in SEED_PAIRS) + " | Total |",
            "|---|" + "---:|" * (len(SEED_PAIRS) + 1),
        ]
    )
    for strategy in STRATEGIES:
        cells = [f"{seed_lookup[(seed, strategy)]['successes']}/6" for seed, _ in SEED_PAIRS]
        row = overall_by[strategy]
        lines.append(
            f"| {DISPLAY_NAMES[strategy]} | " + " | ".join(cells) + f" | {row['successes']}/{row['episodes']} |"
        )
    lines.extend(
        [
            "",
            "## 与 Dense 的逐 seed-task 配对变化",
            "",
            "| Candidate | Both success | Candidate only | Dense only | Both fail | Exact McNemar p | Both-success step delta median |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate in ("version1", "direct_core64"):
        counts = defaultdict(int)
        step_deltas = []
        for row in pairs:
            baseline_success = bool(row["baseline_success"])
            candidate_success = bool(row[f"{candidate}_success"])
            counts[(baseline_success, candidate_success)] += 1
            if baseline_success and candidate_success:
                step_deltas.append(float(row[f"{candidate}_steps_minus_baseline"]))
        candidate_only = counts[(False, True)]
        baseline_only = counts[(True, False)]
        lines.append(
            f"| {DISPLAY_NAMES[candidate]} | {counts[(True, True)]} | {candidate_only} | {baseline_only} | "
            f"{counts[(False, False)]} | {exact_mcnemar_p(candidate_only, baseline_only):.4f} | "
            f"{fmt(statistics.median(step_deltas) if step_deltas else None)} |"
        )
    lines.extend(
        [
            "",
            "## 闭环服务请求时间（次要口径）",
            "",
            "排除每次服务器启动后的第一个请求，但包含不同任务和状态；用于检查长时间运行是否稳定，不作为主加速比。",
            "",
            "| Strategy | Warm requests | Median (s) | P90 (s) | Mean (s) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for strategy in STRATEGIES:
        row = timing_by[strategy]
        lines.append(
            f"| {DISPLAY_NAMES[strategy]} | {row['warm_count']} | {fmt(row['warm_median'], 4)} | "
            f"{fmt(row['warm_p90'], 4)} | {fmt(row['warm_mean'], 4)} |"
        )
    if stable_metrics is not None:
        lines.extend(
            [
                "",
                "## 稳定单 chunk 主计时",
                "",
                "同一已加载模型、相同输入、模式交替、CUDA 同步，compile/CUDA graphs 关闭。",
                "",
                "| Strategy | Median (s) | P90 (s) | Speedup vs Dense | Action cosine | Vision latent cosine | Vision relative-L2 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        dense_median = float(stable_metrics["timing"]["dense"]["median_s"])
        stable_names = {
            "dense": "Dense baseline",
            "version1": "Version1",
            "c_core64_stable_fixed_opt": "Direct Core64 + Stable (V6-B)",
        }
        for mode in stable_metrics["modes"]:
            row = stable_metrics["timing"][mode]
            comparison = stable_metrics["output_metrics"].get(f"dense_vs_{mode}")
            lines.append(
                f"| {stable_names.get(mode, mode)} | {row['median_s']:.6f} | {row['p90_s']:.6f} | "
                f"{dense_median / float(row['median_s']):.3f}x | "
                f"{1.0 if comparison is None else comparison['action']['cosine']:.9f} | "
                f"{1.0 if comparison is None else comparison['vision']['cosine']:.9f} | "
                f"{0.0 if comparison is None else comparison['vision']['relative_l2']:.6f} |"
            )
    if visual_root is not None and visual_root.exists():
        lines.extend(
            [
                "",
                "## 三类代表任务可视化",
                "",
                "这些图来自每个任务的第一个 policy request；对应仿真只运行 1 step 用于采集，不进入 108-episode 成功率。",
                "",
                "| Difficulty / task | G1 mask | G2 mask | G3 mask | Selected Core-block attention | Selected blocks |",
                "|---|---|---|---|---|---|",
            ]
        )
        samples = (
            ("simple", "BananaInBowlTask"),
            ("moderate", "RubiksCubeLeftOfBowlTask"),
            ("complex", "RubiksCubesInBinTask"),
        )
        for difficulty_name, task in samples:
            directory = visual_root / f"{difficulty_name}_{task}"
            selected = json.loads((directory / "selected_core_blocks.json").read_text(encoding="utf-8"))
            links = [f"[G{group}]({directory / f'mask_overlay_g{group}.png'})" for group in range(1, 4)]
            attention = f"[attention]({directory / 'selected_core_blocks_future_spatial_attention.png'})"
            blocks = ", ".join(f"B{block}" for block in sorted(selected["selected_core_blocks"]))
            lines.append(
                f"| {difficulty_name} / {task} | {links[0]} | {links[1]} | {links[2]} | {attention} | {blocks} |"
            )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 不同 policy seed 对应平衡不完全区组中的不同任务对；策略之间在同一 seed-task 单元严格配对。",
            "- policy seed 是生成噪声 seed；仿真初始化由 RoboLab 任务配置控制，并不是独立的 simulator seed。",
            "- 成功样本完成步数有选择偏差，不能脱离成功率单独判断策略优劣。",
            "- 困难任务最多 1500 step；原本更低的官方上限保持不变。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--stable-chunk-metrics", type=Path)
    args = parser.parse_args()
    root = args.experiment_root.expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)

    episodes = load_episode_rows()
    overall = success_summary(episodes, ("strategy",))
    by_difficulty = success_summary(episodes, ("difficulty", "strategy"))
    by_seed = success_summary(episodes, ("policy_seed", "strategy"))
    pairs = paired_rows(episodes)
    timing_samples, timing_summary = load_closed_loop_timings(root)

    stable_metrics = None
    if args.stable_chunk_metrics is not None:
        stable_metrics = json.loads(args.stable_chunk_metrics.read_text(encoding="utf-8"))

    write_csv(root / "episode_results.csv", episodes)
    write_csv(root / "summary_by_strategy.csv", overall)
    write_csv(root / "summary_by_difficulty.csv", by_difficulty)
    write_csv(root / "summary_by_seed.csv", by_seed)
    write_csv(root / "paired_seed_task.csv", pairs)
    write_csv(root / "closed_loop_chunk_timing_samples.csv", timing_samples)
    write_csv(root / "closed_loop_chunk_timing_summary.csv", timing_summary)
    metrics = {
        "schema_version": 1,
        "episode_count": len(episodes),
        "paired_cell_count": len(pairs),
        "overall": overall,
        "by_difficulty": by_difficulty,
        "by_seed": by_seed,
        "closed_loop_chunk_timing": timing_summary,
        "stable_chunk_metrics_path": str(args.stable_chunk_metrics) if args.stable_chunk_metrics else None,
    }
    (root / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "report_cn.md").write_text(
        build_report(
            overall,
            by_difficulty,
            by_seed,
            pairs,
            timing_summary,
            stable_metrics,
            root / "visual_samples",
        ),
        encoding="utf-8",
    )
    print(f"wrote summary for {len(episodes)} episodes to {root}")


if __name__ == "__main__":
    main()
