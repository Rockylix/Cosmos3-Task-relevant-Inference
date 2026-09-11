"""Validate and aggregate three ten-task runs without rerunning inference."""

import argparse
import csv
import itertools
import json
import statistics
from pathlib import Path

import numpy as np


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--toca-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    sources = {"dense": root / "dense", "asi": root / "asi", "toca_future": args.toca_dir.resolve()}
    summary, episodes, manifests = {}, {}, {}
    for mode, source in sources.items():
        manifest = json.loads((source / "manifest.json").read_text())
        manifests[mode] = manifest
        assert json.loads((source / "summary.json").read_text())["status"] == "complete"
        rows = read_rows(source / "simulator/episode_results.jsonl")
        assert len(rows) == 10 and {r["task_name"] for r in rows} == set(manifest["tasks"])
        assert len({r["task_name"] for r in rows}) == 10
        episodes[mode] = {r["task_name"]: r for r in rows}
        configs = list((source / "simulator").glob("*/env_cfg.json"))
        assert len(configs) == 10 and all(json.loads(p.read_text())["seed"] == 0 for p in configs)
        runtime = json.loads((source / "server/runtime.json").read_text())
        for key, value in {
            "policy_seed": 0,
            "deterministic_seed": False,
            "reset_seed_on_prompt": True,
            "compile": False,
            "cuda_graphs": False,
            "shift": 5,
            "guidance": 3,
            "num_steps": 4,
        }.items():
            assert runtime[key] == value, (mode, key, runtime[key], value)
        requests = read_rows(source / "server/requests.jsonl")
        assert [r["request"] for r in requests] == list(range(len(requests)))
        assert all(r["block_calls"] == 224 and r["all_intermediate_finite"] for r in requests)
        prompt_groups = []
        for prompt, group in itertools.groupby(requests, key=lambda r: r["prompt"]):
            group = list(group)
            rng = np.random.default_rng(0)
            assert [r["seed"][0] for r in group] == rng.integers(0, 2**31, len(group)).tolist()
            assert [r["prompt_chunk"] for r in group] == list(range(1, len(group) + 1))
            prompt_groups.append(prompt)
        assert len(prompt_groups) == 10
        if mode == "asi":
            assert all(
                (
                    r["strategy_summary"]["core_token_budget"],
                    r["strategy_summary"]["stable_token_budget"],
                    r["strategy_summary"]["dense_stack_count"],
                    r["strategy_summary"]["sparse_stack_count"],
                )
                == (80, 104, 1, 7)
                for r in requests
            )
        if mode == "toca_future":
            assert all(r["config"]["full_steps"] == [0, 2] and r["config"]["fresh_ratio"] == 0.25 for r in requests)
        successful = [r for r in rows if r["success"]]
        warm = [r["generation_wall_s"] for r in requests if r["prompt_chunk"] > 1]
        summary[mode] = {
            "source_run": str(source),
            "episodes": len(rows),
            "successes": len(successful),
            "success_rate": len(successful) / len(rows),
            "score_mean": statistics.mean(r["score"] for r in rows),
            "steps_mean_all": statistics.mean(r["episode_step"] for r in rows),
            "steps_mean_successful": statistics.mean(r["episode_step"] for r in successful) if successful else None,
            "requests": len(requests),
            "warm_requests": len(warm),
            "observed_chunk_mean_s": statistics.mean(warm),
            "observed_chunk_median_s": statistics.median(warm),
            "observed_chunk_p90_s": float(np.quantile(warm, 0.9)),
            "all_finite": True,
        }
    task_list = manifests["dense"]["tasks"]
    assert all(m["tasks"] == task_list for m in manifests.values())
    combined = []
    for task in task_list:
        row = {"task": task}
        for mode, by_task in episodes.items():
            row.update({f"{mode}_{key}": by_task[task][key] for key in ("success", "score", "episode_step")})
        combined.append(row)
    for mode in ("asi", "toca_future"):
        common = [task for task in task_list if episodes["dense"][task]["success"] and episodes[mode][task]["success"]]
        summary[mode]["paired_both_success"] = {
            "count": len(common),
            "tasks": common,
            "dense_mean_steps": statistics.mean(episodes["dense"][t]["episode_step"] for t in common)
            if common
            else None,
            "strategy_mean_steps": statistics.mean(episodes[mode][t]["episode_step"] for t in common)
            if common
            else None,
        }
    with (root / "closed_loop_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    result = {
        "status": "complete",
        "protocol_checks_passed": True,
        "closed_loop": summary,
        "scope": "Dense/ASI newly run; ToCa closed loop reused from the same-configuration previous run",
    }
    fidelity = root / "fidelity/summary.json"
    if fidelity.exists():
        result["paired_fidelity_and_timing"] = json.loads(fidelity.read_text())
    (root / "comparison_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
