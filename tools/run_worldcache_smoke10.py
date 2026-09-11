"""Frozen ten-task protocol; owned subprocess cleanup; no failed-episode retries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import socket
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROBOLAB = Path("/root/robolab/RoboLab")
PYTHON = Path("/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python")
TOCA = Path("/root/robolab/worktrees/toca-future")
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
VAE = Path(
    "/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/"
    "921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
)
GATE = TOCA / "experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2/dense/server/captures/request_000002/sample.pt"


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_episodes(run):
    path = run / "simulator/episode_results.jsonl"
    rows = []
    if path.exists():
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if i != len(lines) - 1:
                    raise
    tasks = [r["task_name"] for r in rows]
    if len(tasks) != len(set(tasks)) or not set(tasks) <= set(TASKS):
        raise RuntimeError("Duplicate or unexpected episode, no filtering allowed")
    if any(not isinstance(r.get("success"), bool) or r.get("score") is None for r in rows):
        raise RuntimeError("Missing success/score")
    return rows


def stop_owned(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def aggregate(run, phase):
    rows = read_episodes(run)
    paired = [json.loads(p.read_text()) for p in sorted((run / "paired").glob("*.json"))]
    value = {
        "phase": phase,
        "completed": len(rows),
        "expected": 10,
        "success_count": sum(r["success"] for r in rows),
        "success_rate": sum(r["success"] for r in rows) / len(rows) if rows else None,
        "score_mean": statistics.mean(r["score"] for r in rows) if rows else None,
        "paired_tasks": len(paired),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if len(paired) == 10:
        value["timing"] = {
            mode: {
                metric: statistics.mean(p["timing"][mode][metric] for p in paired)
                for metric in ("mean_s", "median_s", "p90_s")
            }
            for mode in ("dense", "worldcache")
        }
        value["speedup"] = value["timing"]["dense"]["median_s"] / value["timing"]["worldcache"]["median_s"]
        value["fidelity"] = {
            key: {
                metric: statistics.mean(p[key][metric] for p in paired) for metric in ("mse", "cosine", "relative_l2")
            }
            for key in ("action", "future_latent", "future_rgb")
        }
    write_json(run / "summary.json", value)
    if rows:
        keys = ["task_name", "success", "score", "episode_step", "reason"]
        with (run / "episodes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows({k: row.get(k) for k in keys} for row in rows)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8028)
    args = parser.parse_args()
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    with socket.socket() as check:
        check.bind(("127.0.0.1", args.port))
    for path in (PYTHON, ROBOLAB / ".venv/bin/python", VAE, GATE):
        if not path.exists():
            raise FileNotFoundError(path)
    source_files = [
        "cosmos_framework/inference/worldcache.py",
        "cosmos_framework/model/generator/omni_mot_model.py",
        "cosmos_framework/scripts/action_policy_server_robolab.py",
        "cosmos_framework/scripts/worldcache_smoke.py",
    ]
    write_json(
        run / "manifest.json",
        {
            "tasks": TASKS,
            "difficulty": {t: "simple" if i < 8 else "moderate" for i, t in enumerate(TASKS)},
            "strategy": "worldcache_dddc_joint",
            "schedule": ["D", "D", "D", "C"],
            "stable_percentile": 0.3,
            "chaotic_percentile": 0.7,
            "n_max": 6,
            "cfg": "independent",
            "simulator_seed": 0,
            "policy_seed": 0,
            "deterministic_seed": False,
            "reset_rng_each_task": True,
            "shift": 5,
            "guidance": 3,
            "steps": 4,
            "compile": False,
            "cuda_graph": False,
            "official_task_step_limits": True,
            "rollouts": 1,
            "retries": 0,
            "reference_toca": json.loads((TOCA / "configs/toca_baseline.json").read_text()),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files},
        },
    )
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="0",
        COSMOS_TRAINING="0",
        LD_LIBRARY_PATH="",
        HF_HOME="/root/cosmos3/cosmos/checkpoints/hf_home",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONPATH=f"{ROOT}:/root/robolab/cosmos-edge-overlay",
        PYTHONUNBUFFERED="1",
        NO_PROXY="localhost,127.0.0.1",
        no_proxy="localhost,127.0.0.1",
        OMNI_KIT_ACCEPT_EULA="Y",
        ACCEPT_EULA="Y",
        PRIVACY_CONSENT="Y",
    )
    common = [
        str(PYTHON),
        "-u",
        "-m",
        "cosmos_framework.scripts.worldcache_smoke",
        "--eval-root",
        str(run),
        "--gate-capture",
        str(GATE),
        "--checkpoint-path",
        str(ROBOLAB / "Cosmos3-Edge-Policy-DROID"),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--no-guardrails",
        "--output-dir",
        str(run / "model_output"),
        "--experiment-overrides",
        f"model.config.tokenizer.vae_path={VAE}",
        "model.config.tokenizer.object_store_credential_path_pretrained=",
        "model.config.tokenizer.bucket_name=",
    ]
    server_cmd = common + ["--eval-phase", "serve"]
    benchmark_cmd = common + ["--eval-phase", "benchmark"]
    sim_cmd = [
        str(ROBOLAB / ".venv/bin/python"),
        "policies/cosmos3/run.py",
        "--remote-host",
        "127.0.0.1",
        "--remote-port",
        str(args.port),
        "--task",
        *TASKS,
        "--num-envs",
        "1",
        "--num-runs",
        "1",
        "--headless",
        "--video-mode",
        "none",
        "--output-folder-name",
        str(run / "simulator"),
    ]
    write_json(run / "commands.json", {"server": server_cmd, "simulator": sim_cmd, "benchmark": benchmark_cmd})
    server = simulator = benchmark = None
    try:
        aggregate(run, "gpu_gate_and_server_start")
        with (run / "server.log").open("w") as log:
            server = subprocess.Popen(
                server_cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        deadline = time.monotonic() + 600
        while "[worldcache-server] READY" not in (run / "server.log").read_text(errors="replace"):
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Server/GPU gate failed; see server.log")
            time.sleep(2)
        print("[WorldCache] GPU gate passed; starting 10 episodes (8 simple + 2 moderate)", flush=True)
        sim_env = {**env, "PYTHONPATH": f"{ROBOLAB}:/root/robolab/cosmos-edge-overlay"}
        with (run / "simulator.log").open("w") as log:
            simulator = subprocess.Popen(
                sim_cmd, cwd=ROBOLAB, env=sim_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        last = -1
        while simulator.poll() is None:
            if server.poll() is not None:
                raise RuntimeError("Server exited during simulation")
            val = aggregate(run, "closed_loop")
            if val["completed"] != last:
                last = val["completed"]
                score = "—" if val["score_mean"] is None else f"{val['score_mean']:.4f}"
                print(
                    f"[WorldCache] {last}/10 completed | success {val['success_count']}/{last} | score {score}",
                    flush=True,
                )
            time.sleep(5)
        if len(read_episodes(run)) != 10:
            raise RuntimeError(f"Simulation ended with only {len(read_episodes(run))}/10 episodes; no auto-retry")
        aggregate(run, "benchmark")
        stop_owned(server)
        print("[WorldCache] 10 episodes finished; simulator stopped; paired GPU-only warm timing starts", flush=True)
        with (run / "benchmark.log").open("w") as log:
            benchmark = subprocess.Popen(
                benchmark_cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        last = -1
        while benchmark.poll() is None:
            val = aggregate(run, "benchmark")
            if val["paired_tasks"] != last:
                last = val["paired_tasks"]
                print(f"[WorldCache] paired timing {last}/10 tasks", flush=True)
            time.sleep(5)
        if benchmark.returncode != 0:
            raise RuntimeError("Paired benchmark failed; closed-loop results preserved")
        val = aggregate(run, "complete")
        print(json.dumps(val, ensure_ascii=False, indent=2), flush=True)
    except BaseException:
        aggregate(run, "stopped_error")
        raise
    finally:
        stop_owned(simulator)
        stop_owned(server)
        stop_owned(benchmark)


if __name__ == "__main__":
    main()
