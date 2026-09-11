"""User-approved two chunk recapture and corrected paired measurement. Episodes unchanged."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from run_worldcache_smoke10 import ROBOLAB, ROOT, aggregate, read_episodes, stop_owned, write_json


def replace_arg(command, flag, value):
    new = command.copy()
    new[new.index(flag) + 1] = str(value)
    return new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    episodes = run / "simulator/episode_results.jsonl"
    digest = hashlib.sha256(episodes.read_bytes()).hexdigest()
    if len(read_episodes(run)) != 10:
        raise RuntimeError("Ten original episodes are required")
    manifest = json.loads((run / "manifest.json").read_text())
    commands = json.loads((run / "commands.json").read_text())
    recapture = run / "recapture_first2"
    recapture.mkdir(exist_ok=False)
    write_json(
        recapture / "manifest.json",
        {
            **manifest,
            "tasks": manifest["tasks"][:2],
            "purpose": "User approved recapture chunk3 only; not additional success-rate observations",
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
    server_cmd = replace_arg(commands["server"], "--eval-root", recapture)
    server_cmd = replace_arg(server_cmd, "--output-dir", recapture / "model_output")
    server_cmd = replace_arg(server_cmd, "--eval-phase", "capture")
    sim_cmd = commands["simulator"].copy()
    lo, hi = sim_cmd.index("--task") + 1, sim_cmd.index("--num-envs")
    sim_cmd[lo:hi] = manifest["tasks"][:2]
    sim_cmd = replace_arg(sim_cmd, "--output-folder-name", recapture / "simulator")
    write_json(recapture / "commands.json", {"server": server_cmd, "simulator": sim_cmd})
    server = simulator = benchmark = None
    try:
        aggregate(run, "recapturing_two_chunks")
        with (recapture / "server.log").open("w") as log:
            server = subprocess.Popen(
                server_cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        deadline = time.monotonic() + 600
        while "[worldcache-server] READY" not in (recapture / "server.log").read_text(errors="replace"):
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Recapture server failed")
            time.sleep(2)
        with (recapture / "simulator.log").open("w") as log:
            simulator = subprocess.Popen(
                sim_cmd,
                cwd=ROBOLAB,
                env={**env, "PYTHONPATH": f"{ROBOLAB}:/root/robolab/cosmos-edge-overlay"},
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        missing = manifest["tasks"][:2]
        while not all((recapture / "_temporary_inputs" / f"{task}.pt").exists() for task in missing):
            if simulator.poll() is not None or server.poll() is not None:
                raise RuntimeError("Recapture stopped before both third chunks were saved")
            time.sleep(1)
        # End auxiliary rollout prefix as soon as both required inputs are saved.
        stop_owned(simulator)
        stop_owned(server)
        for task in missing:
            source = recapture / "_temporary_inputs" / f"{task}.pt"
            target = run / "_temporary_inputs" / source.name
            if target.exists():
                raise FileExistsError(target)
            shutil.move(str(source), str(target))
        old = run / "invalid_fp32_metrics"
        old.mkdir(exist_ok=False)
        for path in (run / "paired").glob("*.json"):
            shutil.move(str(path), str(old / path.name))
        write_json(
            run / "benchmark_metric_revision.json",
            {
                "reason": "FP64 output metrics; first two inputs recaptured with user approval; repeat all timings on corrected inputs",
                "closed_loop_sha256": digest,
                "recaptured_tasks": missing,
                "source_sha256": {
                    p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                    for p in (
                        "cosmos_framework/inference/worldcache.py",
                        "cosmos_framework/scripts/worldcache_smoke.py",
                    )
                },
            },
        )
        print("[WorldCache] two chunk3 inputs recaptured; original success-rate results unchanged", flush=True)
        with (run / "benchmark_fp64.log").open("x") as log:
            benchmark = subprocess.Popen(
                commands["benchmark"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        last = -1
        while benchmark.poll() is None:
            result = aggregate(run, "benchmark_fp64")
            if result["paired_tasks"] != last:
                last = result["paired_tasks"]
                print(f"[WorldCache] success remains 5/10; corrected paired timing/fidelity {last}/10", flush=True)
            time.sleep(5)
        if benchmark.returncode:
            raise RuntimeError("Corrected benchmark failed")
        if hashlib.sha256(episodes.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Original episode results changed")
        print(json.dumps(aggregate(run, "complete"), ensure_ascii=False, indent=2), flush=True)
    finally:
        stop_owned(simulator)
        stop_owned(server)
        stop_owned(benchmark)


if __name__ == "__main__":
    main()
