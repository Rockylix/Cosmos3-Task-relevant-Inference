"""Three user-confirmed tasks, one episode each, no episode retries or videos."""

from __future__ import annotations

import argparse
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
TASKS = ["BananaInBowlTask", "BananaOnPlateTask", "ButterAboveRaisinTask"]
VAE = Path(
    "/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/"
    "921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def stop_owned(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def read_rows(path):
    if not path.exists():
        return []
    content = path.read_text()
    lines = content.splitlines()
    if content and not content.endswith("\n"):
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def aggregate(run, phase):
    episodes = read_rows(run / "simulator/episode_results.jsonl")
    if len({row["task_name"] for row in episodes}) != len(episodes):
        raise RuntimeError("Duplicate episodes; cannot silently filter results")
    if any(row["task_name"] not in TASKS or not isinstance(row["success"], bool) for row in episodes):
        raise RuntimeError("Unexpected episode")
    requests = read_rows(run / "requests.jsonl")
    if any(
        (row["dense_stack_count"], row["sparse_stack_count"], row["cache_evaluations"]) != (1, 7, 4) for row in requests
    ):
        raise RuntimeError("Unexpected execution schedule")
    value = {
        "phase": phase,
        "completed": len(episodes),
        "expected": 3,
        "success_count": sum(row["success"] for row in episodes),
        "success_rate": statistics.mean(row["success"] for row in episodes) if episodes else None,
        "score_mean": statistics.mean(row["score"] for row in episodes) if episodes else None,
        "chunks": len(requests),
        "episodes": [
            {key: row.get(key) for key in ("task_name", "success", "score", "episode_step", "reason")}
            for row in episodes
        ],
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timing_note": "Generation measured with simulator sharing GPU; excludes reporting I/O, not an isolated benchmark",
    }
    write_json(run / "summary.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8029)
    parser.add_argument("--task-core-top7", action="store_true")
    args = parser.parse_args()
    with socket.socket() as check:
        check.bind(("127.0.0.1", args.port))
    for path in (PYTHON, ROBOLAB / ".venv/bin/python", VAE, ROBOLAB / "Cosmos3-Edge-Policy-DROID"):
        if not path.exists():
            raise FileNotFoundError(path)
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    sources = [
        "cosmos_framework/scripts/robolab_version1.py",
        "cosmos_framework/inference/asi_velocity_cache.py",
        "cosmos_framework/inference/task_core_layers.py",
        "cosmos_framework/scripts/asi_velocity_cache_smoke.py",
        "cosmos_framework/scripts/asi_velocity_cache_chunk.py",
        "tools/run_asi_velocity_cache_smoke3.py",
    ]
    write_json(
        run / "manifest.json",
        {
            "tasks": TASKS,
            "simulator_seed": 0,
            "policy_seed": 0,
            "deterministic_seed": False,
            "reset_policy_rng_each_task": True,
            "steps": 4,
            "guidance": 3,
            "shift": 5,
            "compile": False,
            "cuda_graphs": False,
            "dense_forwards": 1,
            "sparse_forwards": 7,
            "cache_source": "step0 ASI guided velocity; NOT fully dense CFG velocity",
            "cache_scope": "removed future positions, after CFG and before native UniPC; reset every chunk",
            "mask_refresh": "every chunk",
            "core": 80,
            "stable": 104,
            "topk_blocks": 7 if args.task_core_top7 else 6,
            "task_core_top7": args.task_core_top7,
            "core_layer_refresh": "task first chunk only" if args.task_core_top7 else "every chunk",
            "stable_source": "all 28 layers, current chunk",
            "rollouts_each": 1,
            "episode_retries": 0,
            "task_step_limits": "official",
            "video_mode": "none",
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        },
    )
    env = dict(os.environ)
    env.pop("LD_PRELOAD", None)
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
    server_cmd = [
        str(PYTHON),
        "-u",
        "-m",
        "cosmos_framework.scripts.asi_velocity_cache_smoke",
        "--eval-root",
        str(run),
        "--checkpoint-path",
        str(ROBOLAB / "Cosmos3-Edge-Policy-DROID"),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--no-guardrails",
        "--output-dir",
        str(run / "model_output"),
        "--seed",
        "0",
        "--shift",
        "5",
        "--num-steps",
        "4",
        "--guidance",
        "3",
        "--experiment-overrides",
        f"model.config.tokenizer.vae_path={VAE}",
        "model.config.tokenizer.object_store_credential_path_pretrained=",
        "model.config.tokenizer.bucket_name=",
    ]
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
    if args.task_core_top7:
        server_cmd.insert(4, "--task-core-top7")
    write_json(run / "commands.json", {"server": server_cmd, "simulator": sim_cmd})
    server = simulator = None
    try:
        aggregate(run, "gate_and_server_start")
        with (run / "server.log").open("w") as stream:
            server = subprocess.Popen(
                server_cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )
        deadline = time.monotonic() + 600
        while "[asi-velocity-server] READY" not in (run / "server.log").read_text(errors="replace"):
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Server/GPU gate failed; inspect server.log")
            time.sleep(2)
        print("[ASI+VC] GPU gate passed; starting the three confirmed tasks", flush=True)
        sim_env = {**env, "PYTHONPATH": f"{ROBOLAB}:/root/robolab/cosmos-edge-overlay"}
        with (run / "simulator.log").open("w") as stream:
            simulator = subprocess.Popen(
                sim_cmd, cwd=ROBOLAB, env=sim_env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )
        previous = None
        while simulator.poll() is None:
            if server.poll() is not None:
                raise RuntimeError("Policy server exited during simulation")
            result = aggregate(run, "closed_loop")
            status = (result["completed"], result["chunks"])
            if previous != status:
                previous = status
                print(
                    f"[ASI+VC] completed={result['completed']}/3 success={result['success_count']}/{result['completed']} score={result['score_mean']} chunks={result['chunks']}",
                    flush=True,
                )
            time.sleep(5)
        result = aggregate(run, "complete")
        if result["completed"] != 3 or simulator.returncode != 0:
            raise RuntimeError("Simulation incomplete; no automatic episode retry")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except BaseException:
        aggregate(run, "stopped_error")
        raise
    finally:
        stop_owned(simulator)
        stop_owned(server)


if __name__ == "__main__":
    main()
