#!/usr/bin/env python3
"""Run the 6-seed x 6-task x 3-strategy RoboLab closed-loop evaluation.

The runner is resumable at the task level: RoboLab's append-only result file is
reused and completed seed/task cells are skipped.  One policy server is started
per (strategy, policy seed), then the six tasks assigned to that seed are run in
one Isaac Sim process.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

EDGE_ROOT = Path("/root/robolab/cosmos-framework-edge-version1")
ROBOLAB_ROOT = Path("/root/robolab/RoboLab")
EXPERIMENT_ROOT = Path(
    "/root/robolab/cosmos-framework-edge-version1/experiments/preliminary/sparsity/evaluation/"
    "direct_core64_baseline_v1_bibd_6seeds_6tasks_v1"
)
CHECKPOINT = ROBOLAB_ROOT / "Cosmos3-Edge-Policy-DROID"
COSMOS_PYTHON = Path("/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python")
ROBOLAB_PYTHON = ROBOLAB_ROOT / ".venv/bin/python"
PORT = 8000

SEED_PAIRS: tuple[tuple[int, str], ...] = (
    (9573, "AB"),
    (7685, "AC"),
    (5732, "AD"),
    (7640, "BC"),
    (4701, "BD"),
    (158, "CD"),
)

TASKS: dict[str, dict[str, str]] = {
    "simple": {
        "A": "BananaInBowlTask",
        "B": "BananaOnPlateTask",
        "C": "RubiksCubeTask",
        "D": "YogurtInBowlTask",
    },
    "moderate": {
        "A": "RubiksCubeLeftOfBowlTask",
        "B": "RubiksCubeInFrontOfBowlTask",
        "C": "Stack3RubiksCubeTask",
        "D": "UnstackRubiksCubeTask",
    },
    "complex": {
        "A": "RubiksCubesInBinTask",
        "B": "BlockStackingSpecifiedOrderTask",
        "C": "ReorientAllMugsTask",
        "D": "FruitsOnPlate3Task",
    },
}

STRATEGIES: dict[str, dict[str, str]] = {
    "baseline": {
        "module": "cosmos_framework.scripts.action_policy_server_robolab_v5_3_acd_packed_kernel",
        "mode": "dense",
    },
    "version1": {
        "module": "cosmos_framework.scripts.action_policy_server_robolab_step0_fixed_roi_velocity_cache",
        "mode": "",
    },
    "direct_core64": {
        "module": "cosmos_framework.scripts.action_policy_server_robolab_v5_3_acd_packed_kernel",
        "mode": "c_core64_stable_fixed_b0_sparse",
    },
}


def tasks_for_pair(pair: str) -> list[str]:
    return [TASKS[difficulty][letter] for difficulty in TASKS for letter in pair]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def completed_tasks(output_name: str) -> set[str]:
    rows = read_jsonl(ROBOLAB_ROOT / "output" / output_name / "episode_results.jsonl")
    return {str(row["task_name"]) for row in rows if int(row.get("run", 0)) == 0}


def wait_for_server(process: subprocess.Popen[Any], log_path: Path, timeout_s: float = 900.0) -> None:
    deadline = time.monotonic() + timeout_s
    next_status = time.monotonic()
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            tail = ""
            if log_path.exists():
                tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
            raise RuntimeError(f"policy server exited with code {code}\n{tail}")
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=1.0):
                return
        except OSError:
            pass
        if time.monotonic() >= next_status:
            print(f"[runner] waiting for policy server; log={log_path}", flush=True)
            next_status = time.monotonic() + 30.0
        time.sleep(2.0)
    raise TimeoutError(f"policy server was not ready within {timeout_s:.0f}s; log={log_path}")


def stop_process_group(process: subprocess.Popen[Any], name: str) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=45)
        return
    except subprocess.TimeoutExpired:
        print(f"[runner] {name} ignored SIGINT; sending SIGTERM", flush=True)
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        print(f"[runner] {name} ignored SIGTERM; sending SIGKILL", flush=True)
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def policy_server_command(strategy: str, seed: int, server_dir: Path) -> list[str]:
    spec = STRATEGIES[strategy]
    command = [
        str(COSMOS_PYTHON),
        "-m",
        spec["module"],
        "--checkpoint-path",
        str(CHECKPOINT),
        "--format-prompt-as-json",
        "True",
        "--no-guardrails",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--guidance",
        "3",
        "--num-steps",
        "4",
        "--shift",
        "5",
        "--seed",
        str(seed),
        "--deterministic-seed",
        "--intervention-output-dir",
        str(server_dir),
    ]
    if spec["mode"]:
        command.extend(["--ablation-mode", spec["mode"]])
    return command


def simulator_command(tasks: list[str], output_name: str) -> list[str]:
    return [
        str(ROBOLAB_PYTHON),
        "policies/cosmos3/run.py",
        "--remote-host",
        "127.0.0.1",
        "--remote-port",
        str(PORT),
        "--num-envs",
        "1",
        "--num-runs",
        "1",
        "--task",
        *tasks,
        "--max-episode-steps",
        "1500",
        "--output-folder-name",
        output_name,
        "--video-mode",
        "none",
        "--headless",
    ]


def make_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": "/root/cosmos3/cosmos/checkpoints/hf_home",
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": f"/root/robolab/cosmos-edge-overlay:{EDGE_ROOT}",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "OMNI_KIT_ACCEPT_EULA": "Y",
            "PYTHONUNBUFFERED": "1",
            # The policy server is local.  Never route its long-lived WebSocket
            # through the machine's HTTP proxy; proxy resets otherwise look like
            # random mid-episode policy-server failures.
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    return env


def write_runtime_manifest() -> None:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "design": "balanced_incomplete_block_pairs",
        "policy_seeds": [seed for seed, _ in SEED_PAIRS],
        "seed_pairs": [{"seed": seed, "pair": pair, "tasks": tasks_for_pair(pair)} for seed, pair in SEED_PAIRS],
        "task_pools": TASKS,
        "strategies": STRATEGIES,
        "episode_count": len(SEED_PAIRS) * 6 * len(STRATEGIES),
        "max_episode_steps": 1500,
        "seed_semantics": "policy generation seed; RoboLab task configuration controls simulator seed",
        "guidance": 3.0,
        "num_steps": 4,
        "shift": 5.0,
        "torch_compile": False,
        "cuda_graphs": False,
        "video_mode": "none",
        "edge_worktree": str(EDGE_ROOT),
        "robolab_root": str(ROBOLAB_ROOT),
    }
    (EXPERIMENT_ROOT / "manifest.runtime.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def next_attempt_index(job_dir: Path) -> int:
    indices = []
    for path in job_dir.glob("server_attempt_*"):
        try:
            indices.append(int(path.name.rsplit("_", 1)[1]))
        except ValueError:
            pass
    return max(indices, default=0) + 1


def run_job_attempt(strategy: str, seed: int, pair: str, attempt: int) -> None:
    tasks = tasks_for_pair(pair)
    output_name = f"bibd6x6_{strategy}_seed{seed}_v1"
    done = completed_tasks(output_name)
    if set(tasks).issubset(done):
        print(f"[runner] SKIP complete strategy={strategy} seed={seed} tasks=6/6", flush=True)
        return

    job_dir = EXPERIMENT_ROOT / "closed_loop" / strategy / f"seed_{seed}"
    server_dir = job_dir / f"server_attempt_{attempt:03d}"
    log_dir = EXPERIMENT_ROOT / "logs"
    server_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    server_log = log_dir / f"server_{strategy}_seed{seed}_attempt{attempt:03d}.log"
    simulator_log = log_dir / f"simulator_{strategy}_seed{seed}_attempt{attempt:03d}.log"
    env = make_environment()

    print(
        f"[runner] START strategy={strategy} seed={seed} pair={pair} attempt={attempt} "
        f"remaining={len(set(tasks) - done)}/6",
        flush=True,
    )
    with server_log.open("a", encoding="utf-8") as server_stream:
        server = subprocess.Popen(
            policy_server_command(strategy, seed, server_dir),
            cwd=EDGE_ROOT,
            env=env,
            stdout=server_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        wait_for_server(server, server_log)
        with simulator_log.open("a", encoding="utf-8") as simulator_stream:
            simulator = subprocess.Popen(
                simulator_command(tasks, output_name),
                cwd=ROBOLAB_ROOT,
                env=env,
                stdout=simulator_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        next_status = time.monotonic()
        while simulator.poll() is None:
            if server.poll() is not None:
                stop_process_group(simulator, "simulator")
                raise RuntimeError(f"policy server exited during simulator run with code {server.returncode}")
            if time.monotonic() >= next_status:
                now_done = completed_tasks(output_name)
                aggregate = server_dir / "aggregate.json"
                requests = None
                if aggregate.exists():
                    try:
                        requests = json.loads(aggregate.read_text(encoding="utf-8")).get("completed_requests")
                    except (OSError, json.JSONDecodeError):
                        pass
                print(
                    f"[runner] RUNNING strategy={strategy} seed={seed} "
                    f"tasks={len(set(tasks) & now_done)}/6 requests={requests}",
                    flush=True,
                )
                next_status = time.monotonic() + 30.0
            time.sleep(2.0)
        if simulator.returncode != 0:
            tail = "\n".join(
                simulator_log.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
            )
            raise RuntimeError(f"simulator exited with code {simulator.returncode}\n{tail}")
        now_done = completed_tasks(output_name)
        if not set(tasks).issubset(now_done):
            raise RuntimeError(
                f"simulator exited successfully but results are incomplete: {len(set(tasks) & now_done)}/6"
            )
        print(f"[runner] DONE strategy={strategy} seed={seed} tasks=6/6", flush=True)
    finally:
        stop_process_group(server, "policy server")


def run_job(strategy: str, seed: int, pair: str, max_attempts: int = 3) -> None:
    job_dir = EXPERIMENT_ROOT / "closed_loop" / strategy / f"seed_{seed}"
    first_attempt = next_attempt_index(job_dir)
    for offset in range(max_attempts):
        attempt = first_attempt + offset
        try:
            run_job_attempt(strategy, seed, pair, attempt)
            return
        except RuntimeError as exc:
            if offset + 1 >= max_attempts:
                raise
            print(
                f"[runner] RETRY strategy={strategy} seed={seed} "
                f"after recoverable attempt failure: {exc}",
                flush=True,
            )


def job_order() -> list[tuple[str, int, str]]:
    names = list(STRATEGIES)
    jobs = []
    for index, (seed, pair) in enumerate(SEED_PAIRS):
        rotated = names[index % len(names) :] + names[: index % len(names)]
        jobs.extend((strategy, seed, pair) for strategy in rotated)
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=[*STRATEGIES, "all"], default="all")
    parser.add_argument("--seed", type=int, choices=[seed for seed, _ in SEED_PAIRS])
    parser.add_argument("--list", action="store_true", help="Print the resolved job matrix and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_runtime_manifest()
    jobs = job_order()
    if args.strategy != "all":
        jobs = [job for job in jobs if job[0] == args.strategy]
    if args.seed is not None:
        jobs = [job for job in jobs if job[1] == args.seed]
    if args.list:
        for strategy, seed, pair in jobs:
            print(strategy, seed, pair, *tasks_for_pair(pair))
        return
    for strategy, seed, pair in jobs:
        run_job(strategy, seed, pair)
    print("[runner] ALL REQUESTED JOBS COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[runner] interrupted", file=sys.stderr, flush=True)
        raise SystemExit(130)
