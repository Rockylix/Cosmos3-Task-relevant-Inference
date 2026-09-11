"""Run one episode of each of the established ten smoke tasks, without videos.

Only child processes created here are stopped. No environment/asset installation,
task retries, outcome filtering, or baseline-result substitution is performed.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

TASKS = [
    "BananaInBowlTask",
    "BananaOnPlateTask",
    "ButterAboveRaisinTask",
    "BowlStackingLeftOnRightTask",
    "GrabABagelTask",
    "GrabAFruitTask",
    "LargerObjectRaisinBoxInBinTask",
    "MustardInLeftBinTask",
    "PickGlassesTask",
    "RubiksCubeTask",
]
EDGE_PYTHON = Path("/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python")
ROBOLAB = Path("/root/robolab/RoboLab")
VAE = Path(
    "/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
)


def records(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    result = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return result


def stop_owned(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8017)
    parser.add_argument("--attention-backend", choices=("reference", "joint"), default="reference")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    run = args.run_dir.resolve()
    if run.exists():
        raise FileExistsError(f"Refusing to overwrite {run}")
    for path in (EDGE_PYTHON, ROBOLAB / ".venv/bin/python", ROBOLAB / "Cosmos3-Edge-Policy-DROID", VAE):
        if not path.exists():
            raise FileNotFoundError(path)
    with socket.socket() as check:
        check.bind(("127.0.0.1", args.port))
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="0",
        COSMOS_TRAINING="0",
        HF_HOME="/root/cosmos3/cosmos/checkpoints/hf_home",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        LD_LIBRARY_PATH="",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=f"{repo}:/root/robolab/cosmos-edge-overlay",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
        OMNI_KIT_ACCEPT_EULA="Y",
        ACCEPT_EULA="Y",
        PRIVACY_CONSENT="Y",
    )
    run.mkdir(parents=True)
    server_cmd = [
        str(EDGE_PYTHON),
        "-m",
        "cosmos_framework.scripts.action_policy_server_robolab_toca_future",
        "--checkpoint-path",
        str(ROBOLAB / "Cosmos3-Edge-Policy-DROID"),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--seed",
        "0",
        "--no-deterministic-seed",
        "--guidance",
        "3",
        "--num-steps",
        "4",
        "--shift",
        "5",
        "--format-prompt-as-json",
        "True",
        "--no-guardrails",
        "--output-dir",
        str(run / "model_output"),
        "--toca-output-dir",
        str(run / "server"),
        "--toca-attention-backend",
        args.attention_backend,
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
    manifest = {
        "strategy": "toca-future",
        "source": str(repo),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip(),
        "base_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "tasks": TASKS,
        "episodes_per_task": 1,
        "simulator_seed": 0,
        "simulator_seed_source": "runner.create_env omits seed; runtime.create_env defaults to 0",
        "policy_seed": 0,
        "deterministic_seed": False,
        "reset_policy_seed_on_prompt": True,
        "full_steps": [0, 2],
        "fresh_ratio": 0.25,
        "layer_slope": 0.5,
        "attention_backend": args.attention_backend,
        "torch_compile": False,
        "cuda_graphs": False,
        "video_mode": "none",
        "step_limits": "official per-task; no additional cap",
        "server_command": server_cmd,
        "simulator_command": sim_cmd,
        "pythonpath": env["PYTHONPATH"],
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    simulator, server = None, None
    episode_file = run / "simulator/episode_results.jsonl"
    status = "failed"
    try:
        with (run / "server.log").open("w") as server_log, (run / "simulator.log").open("w") as sim_log:
            server = subprocess.Popen(
                server_cmd, cwd=repo, env=env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True
            )
            print(f"[setup] server PID={server.pid}; logs={run}", flush=True)
            for _ in range(300):
                if server.poll() is not None:
                    raise RuntimeError(f"Policy server exited {server.returncode}; inspect server.log")
                try:
                    with socket.create_connection(("127.0.0.1", args.port), timeout=1):
                        break
                except OSError:
                    time.sleep(2)
            else:
                raise TimeoutError("Policy server did not become ready in 600 seconds")
            sim_env = env.copy()
            sim_env["PYTHONPATH"] = f"{ROBOLAB}:/root/robolab/cosmos-edge-overlay"
            simulator = subprocess.Popen(
                sim_cmd, cwd=ROBOLAB, env=sim_env, stdout=sim_log, stderr=subprocess.STDOUT, start_new_session=True
            )
            print(f"[running] simulator PID={simulator.pid}; 10 tasks, simulator seed=0, policy seed=0", flush=True)
            previous = -1
            while simulator.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError("Policy server stopped during evaluation")
                completed = records(episode_file)
                if len(completed) != previous:
                    previous = len(completed)
                    successes = sum(bool(row["success"]) for row in completed)
                    scores = [float(row["score"]) for row in completed if row.get("score") is not None]
                    print(
                        f"[progress] {len(completed)}/10 success={successes}/{len(completed)} "
                        f"score_mean={statistics.mean(scores) if scores else 0:.4f}",
                        flush=True,
                    )
                time.sleep(2)
            completed = records(episode_file)
            if simulator.returncode or len(completed) != 10 or {r["task_name"] for r in completed} != set(TASKS):
                raise RuntimeError(f"Incomplete smoke: exit={simulator.returncode}, episodes={len(completed)}")
            status = "complete"
    finally:
        stop_owned(simulator)
        stop_owned(server)
        completed = records(episode_file)
        requests = records(run / "server/requests.jsonl")
        warm = [r["generation_wall_s"] for r in requests if r["prompt_chunk"] > 1 and not r["validation_preceded"]]
        successes = sum(bool(r["success"]) for r in completed)
        scores = [float(r["score"]) for r in completed if r.get("score") is not None]
        result = {
            "status": status,
            "episodes": len(completed),
            "successes": successes,
            "success_rate": successes / len(completed) if completed else None,
            "score_mean": statistics.mean(scores) if scores else None,
            "requests": len(requests),
            "warm_generation_mean_s": statistics.mean(warm) if warm else None,
            "warm_generation_median_s": statistics.median(warm) if warm else None,
            "warm_generation_p90_s": statistics.quantiles(warm, n=10, method="inclusive")[8] if len(warm) > 1 else None,
            "timing_caveat": "Closed-loop observed generation, includes score and finite checks; not a paired speedup benchmark",
            "all_requests_finite": all(r["all_intermediate_finite"] for r in requests) if requests else None,
        }
        (run / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[failed] {exc}", file=sys.stderr, flush=True)
        raise
