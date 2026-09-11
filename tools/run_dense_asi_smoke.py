"""Sequential paired-protocol ten-task Dense/ASI smoke; no episode retries."""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import time
from pathlib import Path

from run_toca_future_smoke import EDGE_PYTHON, ROBOLAB, TASKS, VAE, records, stop_owned


def run_mode(root, repo, mode, port):
    run = root / mode
    run.mkdir()
    source = Path("/root/robolab/cosmos-framework-edge") if mode == "dense" else repo
    with socket.socket() as check:
        check.bind(("127.0.0.1", port))
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="0",
        COSMOS_TRAINING="0",
        HF_HOME="/root/cosmos3/cosmos/checkpoints/hf_home",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        LD_LIBRARY_PATH="",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=f"{source}:/root/robolab/cosmos-edge-overlay",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
        OMNI_KIT_ACCEPT_EULA="Y",
        ACCEPT_EULA="Y",
        PRIVACY_CONSENT="Y",
    )
    server_cmd = [
        str(EDGE_PYTHON),
        "-P",
        str(repo / "cosmos_framework/scripts/action_policy_server_comparison.py"),
        "--comparison-mode",
        mode,
        "--comparison-output",
        str(run / "server"),
        "--comparison-capture-chunk",
        "3" if mode == "dense" else "0",
        "--checkpoint-path",
        str(ROBOLAB / "Cosmos3-Edge-Policy-DROID"),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
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
        str(port),
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
        "mode": mode,
        "source": str(source),
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "tasks": TASKS,
        "episodes_per_task": 1,
        "simulator_seed": 0,
        "policy_seed": 0,
        "deterministic_seed": False,
        "reset_seed_on_prompt": True,
        "shift": 5,
        "guidance": 3,
        "num_steps": 4,
        "compile": False,
        "cuda_graphs": False,
        "video_mode": "none",
        "step_caps": "official per-task",
        "server_command": server_cmd,
        "simulator_command": sim_cmd,
        "pythonpath": env["PYTHONPATH"],
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    server = simulator = None
    status = "failed"
    try:
        with (run / "server.log").open("w") as slog, (run / "simulator.log").open("w") as ilog:
            server = subprocess.Popen(
                server_cmd, cwd=source, env=env, stdout=slog, stderr=subprocess.STDOUT, start_new_session=True
            )
            print(f"[{mode}] server pid={server.pid} logs={run}", flush=True)
            for _ in range(300):
                if server.poll() is not None:
                    raise RuntimeError(f"{mode} server failed; inspect {run}/server.log")
                if "[comparison] READY" in (run / "server.log").read_text():
                    # Websocket listener starts immediately after this line.
                    time.sleep(2)
                    break
                time.sleep(2)
            else:
                raise TimeoutError("Server readiness timeout")
            sim_env = env.copy()
            sim_env["PYTHONPATH"] = f"{ROBOLAB}:/root/robolab/cosmos-edge-overlay"
            simulator = subprocess.Popen(
                sim_cmd, cwd=ROBOLAB, env=sim_env, stdout=ilog, stderr=subprocess.STDOUT, start_new_session=True
            )
            last = -1
            while simulator.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError("Policy server exited mid-run")
                episodes = records(run / "simulator/episode_results.jsonl")
                if len(episodes) != last:
                    last = len(episodes)
                    score = statistics.mean(float(r["score"]) for r in episodes) if episodes else 0
                    print(
                        f"[{mode}] {last}/10 success={sum(bool(r['success']) for r in episodes)}/{last} score={score:.4f}",
                        flush=True,
                    )
                time.sleep(2)
            episodes = records(run / "simulator/episode_results.jsonl")
            if simulator.returncode or len(episodes) != 10 or {r["task_name"] for r in episodes} != set(TASKS):
                raise RuntimeError(f"Incomplete {mode}: exit={simulator.returncode}, episodes={len(episodes)}")
            status = "complete"
    finally:
        stop_owned(simulator)
        stop_owned(server)
        episodes = records(run / "simulator/episode_results.jsonl")
        requests = records(run / "server/requests.jsonl")
        warm = [r["generation_wall_s"] for r in requests if r["prompt_chunk"] > 1]
        summary = {
            "status": status,
            "mode": mode,
            "episodes": len(episodes),
            "successes": sum(bool(r["success"]) for r in episodes),
            "score_mean": statistics.mean(float(r["score"]) for r in episodes) if episodes else None,
            "requests": len(requests),
            "warm_requests": len(warm),
            "warm_generation_mean_s": statistics.mean(warm) if warm else None,
            "warm_generation_median_s": statistics.median(warm) if warm else None,
            "warm_generation_p90_s": statistics.quantiles(warm, n=10, method="inclusive")[8] if len(warm) > 1 else None,
            "all_finite": all(r["all_intermediate_finite"] for r in requests) if requests else None,
            "timing_scope": "Closed-loop co-resident generation including validation; not paired identical-input speedup",
        }
        (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8018)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    for mode in ("dense", "asi"):
        run_mode(root, repo, mode, args.port)


if __name__ == "__main__":
    main()
