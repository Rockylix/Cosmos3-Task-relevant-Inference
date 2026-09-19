"""Matched Dense/SpecPrune ten-task closed loop; no retry/filtering or videos."""

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROBOLAB = Path("/root/robolab/RoboLab")
PYTHON = "/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python"
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


def rows(path):
    if not path.exists():
        return []
    text = path.read_text()
    return [json.loads(line) for line in text.splitlines(keepends=True) if line.endswith("\n") and line.strip()]


def stop_owned(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def summary(run):
    result = {}
    for strategy in ("dense", "specprune"):
        records = rows(run / strategy / "simulator/episode_results.jsonl")
        if len(records) > 10 or len({r["task_name"] for r in records}) != len(records):
            raise ValueError("Duplicate/unexpected episodes")
        result[strategy] = {
            "completed": len(records),
            "success": sum(r["success"] for r in records),
            "mean_score": sum(r["score"] for r in records) / len(records) if records else None,
            "episodes": records,
            "chunks": len(rows(run / strategy / "requests.jsonl")),
        }
    temp = run / "status.tmp"
    temp.write_text(json.dumps(result, indent=2, allow_nan=False))
    temp.replace(run / "status.json")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8041)
    parser.add_argument(
        "--dense-from", type=Path,
        help="User-authorized fresh Sparse attempt using completed Dense results from another run",
    )
    parser.add_argument(
        "--resume-sparse",
        action="store_true",
        help="Only after all ten Dense episodes; refuse existing Sparse outcomes",
    )
    parser.add_argument("--disable-dynamic", action="store_true")
    parser.add_argument("--dynamic-source", choices=["initial_noise", "observation"], default="observation")
    args = parser.parse_args()
    if args.dense_from and args.resume_sparse:
        parser.error("--dense-from and --resume-sparse are mutually exclusive")
    gate = json.loads(args.gate.read_text())
    if gate.get("dynamic_enabled", True) != (not args.disable_dynamic):
        raise RuntimeError("Gate and experiment Dynamic configurations differ")
    if gate.get("dynamic_source", "initial_noise") != args.dynamic_source:
        raise RuntimeError("Gate and experiment Dynamic sources differ")
    for name, expected in gate.get("source_sha256", {}).items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Inference source changed since GPU gate")
    if any(value["mse"] != 0 for value in gate["full_keep"].values()):
        raise RuntimeError("Full keep gate is not exact; no closed loop permitted")
    with socket.socket() as check:
        check.bind(("127.0.0.1", args.port))
    run = args.output.resolve()
    if args.dense_from:
        prior = args.dense_from.resolve()
        if rows(prior / "specprune/simulator/episode_results.jsonl") or rows(prior / "specprune/requests.jsonl"):
            raise RuntimeError("This restart is only authorized for the aborted first request")
        if gate.get("native_shapes", {}).get("vision") != [1, 48, 9, 33, 40]:
            raise RuntimeError("Restart requires the corrected native-shape GPU gate")
        done = rows(prior / "dense/simulator/episode_results.jsonl")
        original_dense = json.loads((prior / "manifest.json").read_text())
        if len(done) != 10 or {r["task_name"] for r in done} != set(TASKS):
            raise RuntimeError("Expected ten unique completed Dense tasks")
        expected = dict(tasks=TASKS, simulator_seed=0, policy_seed=0, steps=4, guidance=3, shift=5,
                        compile=False, cuda_graphs=False, rollouts_per_task=1)
        if any(original_dense.get(k) != v for k, v in expected.items()):
            raise RuntimeError("Prior Dense protocol mismatch")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if original_dense["git_head"] != head:
            raise RuntimeError("Dense source HEAD changed")
        subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "cosmos_framework/model"], cwd=ROOT, check=True)
    if args.resume_sparse:
        done = rows(run / "dense/simulator/episode_results.jsonl")
        if len(done) != 10 or {r["task_name"] for r in done} != set(TASKS):
            raise RuntimeError("Resume requires all ten unique Dense results")
        if (run / "specprune").exists():
            raise RuntimeError("Refusing to replace any existing Sparse attempt")
        original = json.loads((run / "manifest.json").read_text())
        if original["tasks"] != TASKS or original["simulator_seed"] != 0 or original["policy_seed"] != 0:
            raise RuntimeError("Dense protocol does not match")
    else:
        run.mkdir(parents=True, exist_ok=False)
    if args.dense_from:
        (run / "dense").symlink_to(prior / "dense", target_is_directory=True)
    files = [
        "cosmos_framework/inference/specprune_exit_metrics.py",
        "cosmos_framework/inference/specprune_observation_exit.py",
        "cosmos_framework/inference/specprune_exit_plan.py",
        "cosmos_framework/inference/specprune_future.py",
        "cosmos_framework/inference/future_instruction_attention.py",
        "cosmos_framework/inference/specprune_observation.py",
        "cosmos_framework/scripts/specprune_exit_server.py",
        "tools/run_specprune_exit_smoke10.py",
    ]
    manifest = {
        "dynamic_enabled": not args.disable_dynamic,
        "dynamic_source": args.dynamic_source,
        "tasks": TASKS,
        "simulator_seed": 0,
        "policy_seed": 0,
        "rollouts_per_task": 1,
        "strategies": ["dense", "specprune"],
        "steps": 4,
        "guidance": 3,
        "shift": 5,
        "compile": False,
        "cuda_graphs": False,
        "video_mode": "none",
        "episode_retries": 0,
        "task_step_limits": "official",
        "gate": str(args.gate.resolve()),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
    }
    if args.dense_from:
        manifest.update(
            phase="authorized_sparse_restart_after_output_shape_fix",
            dense_manifest=str(prior / "manifest.json"),
            dense_results_reused=True,
            previous_aborted_attempt=str(prior / "specprune"),
            reason="User approved restart; prior attempt returned no action and completed no Sparse episode",
        )
        manifest["dense_episode_results_sha256"] = hashlib.sha256(
            (prior / "dense/simulator/episode_results.jsonl").read_bytes()
        ).hexdigest()
    if args.resume_sparse:
        manifest["phase"] = "sparse_after_completed_dense"
        manifest["dense_manifest"] = str(run / "manifest.json")
        manifest["reason"] = (
            "Correct dynamic-floor importance stopping before any Sparse episode; Dense native generation unchanged"
        )
        # Only selection-only code and the coordinator changed. Dense never calls this selector.
        for name, sha in original["source_sha256"].items():
            if name not in ("cosmos_framework/inference/specprune_exit_plan.py", "tools/run_specprune_exit_smoke10.py"):
                if manifest["source_sha256"][name] != sha:
                    raise RuntimeError(f"Unexpected inference change since Dense: {name}")
        (run / "sparse_manifest.json").write_text(json.dumps(manifest, indent=2))
    else:
        (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "COSMOS_TRAINING": "0",
        "LD_LIBRARY_PATH": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONPATH": f"{ROOT}:/root/robolab/cosmos-edge-overlay",
        "PYTHONUNBUFFERED": "1",
        "MPLCONFIGDIR": "/tmp/specprune-mpl",
        "NO_PROXY": "localhost,127.0.0.1",
        "no_proxy": "localhost,127.0.0.1",
        "OMNI_KIT_ACCEPT_EULA": "Y",
        "ACCEPT_EULA": "Y",
        "PRIVACY_CONSENT": "Y",
    }
    env.pop("LD_PRELOAD", None)
    for strategy in ("specprune",) if (args.resume_sparse or args.dense_from) else ("dense", "specprune"):
        folder = run / strategy
        folder.mkdir()
        server_cmd = [
            PYTHON,
            "-u",
            "-m",
            "cosmos_framework.scripts.specprune_exit_server",
            "--strategy",
            strategy,
            "--run",
            str(folder),
            "--port",
            str(args.port),
            "--dynamic-source",
            args.dynamic_source,
        ]
        if args.disable_dynamic:
            server_cmd.append("--disable-dynamic")
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
            str(folder / "simulator"),
        ]
        (folder / "commands.json").write_text(json.dumps({"server": server_cmd, "simulator": sim_cmd}, indent=2))
        server = sim = None
        try:
            with (folder / "server.log").open("w") as stream:
                server = subprocess.Popen(
                    server_cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
                )
            deadline = time.monotonic() + 600
            while "SPECPRUNE_SERVICE_READY" not in (folder / "server.log").read_text(errors="replace"):
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Server failed to start; inspect server.log")
                time.sleep(2)
            sim_env = {**env, "PYTHONPATH": f"{ROBOLAB}:/root/robolab/cosmos-edge-overlay"}
            with (folder / "simulator.log").open("w") as stream:
                sim = subprocess.Popen(
                    sim_cmd, cwd=ROBOLAB, env=sim_env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
                )
            previous = None
            while sim.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError("Server exited during episode")
                state = summary(run)[strategy]
                key = (state["completed"], state["chunks"])
                if key != previous:
                    print(
                        f"{strategy}: completed={state['completed']}/10 success={state['success']} score={state['mean_score']} chunks={state['chunks']}",
                        flush=True,
                    )
                    previous = key
                time.sleep(5)
            state = summary(run)[strategy]
            if sim.returncode or state["completed"] != 10:
                raise RuntimeError("Simulation incomplete; no automatic retry")
        finally:
            stop_owned(sim)
            stop_owned(server)
    print("ALL_TWENTY_EPISODES_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
