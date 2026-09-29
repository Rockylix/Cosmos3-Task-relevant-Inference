"""Ten fixed RoboLab tasks using the production optimized ASI server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

from tools.run_asi_smoke10 import PYTHON, ROBOLAB, ROOT, TASKS, VAE, read_episodes, stop_owned, write_json


def server(args):
    from cosmos_framework.inference.common.init import init_script

    init_script()
    import numpy as np
    import torch

    from cosmos_framework.scripts.action_policy_server_robolab import _load_openpi_websocket_policy_server
    from cosmos_framework.scripts.action_policy_server_robolab_version1 import Version1PolicyService, Version1ServerArgs

    metadata = json.loads((ROBOLAB / "robolab/tasks/_metadata/task_metadata.json").read_text())
    task_for_prompt = {r["instruction"]: r["task_name"] for r in metadata if r["task_name"] in TASKS}

    class Service(Version1PolicyService):
        def __init__(self, config):
            self.task, self.chunk = None, 0
            super().__init__(config)
            generate = self.model.generate_samples_from_batch

            def measured(*positional, **kwargs):
                torch.cuda.synchronize()
                started = time.perf_counter()
                samples = generate(*positional, **kwargs)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                for key in ("action", "vision"):
                    if not torch.isfinite(samples[key][0]).all().item():
                        raise FloatingPointError(key)
                row = dict(
                    task=self.task,
                    chunk=self.chunk,
                    seed=kwargs["seed"],
                    generation_wrapper_s=elapsed,
                    all_final_finite=True,
                    execution=args.execution,
                )
                with (args.output / "requests.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                return samples

            self.model.generate_samples_from_batch = measured

        def infer(self, observation):
            task = task_for_prompt[observation["prompt"]]
            if self.task != task:
                self.task, self.chunk = task, 0
                self._rng = np.random.default_rng(self.cfg.seed)
            self.chunk += 1
            return super().infer(observation)

    service = Service(
        Version1ServerArgs(
            checkpoint_path=str(ROBOLAB / "Cosmos3-Edge-Policy-DROID"),
            asi_execution=args.execution,
            host="127.0.0.1",
            port=args.port,
            seed=0,
            deterministic_seed=False,
            num_steps=4,
            guidance=3,
            shift=5,
            guardrails=False,
            format_prompt_as_json=True,
            decode_video=False,
            output_dir=args.output / "model_output",
            experiment_overrides=[
                f"model.config.tokenizer.vae_path={VAE}",
                "model.config.tokenizer.object_store_credential_path_pretrained=",
                "model.config.tokenizer.bucket_name=",
            ],
        )
    )
    write_json(
        args.output / "runtime.json",
        dict(
            python=sys.executable,
            torch=torch.__version__,
            gpu=torch.cuda.get_device_name(),
            source=str(ROOT),
            compile=service.setup_args.use_torch_compile,
            cuda_graph=service.setup_args.use_cuda_graphs,
            compile_dynamic=service.setup_args.compile_dynamic,
        ),
    )
    print("OPTIMIZED_SMOKE_READY", flush=True)
    _load_openpi_websocket_policy_server()(
        policy=service, host="127.0.0.1", port=args.port, metadata={}
    ).serve_forever()


def aggregate(run, phase):
    rows = read_episodes(run)
    value = dict(
        phase=phase,
        completed=len(rows),
        expected=10,
        successes=sum(r["success"] for r in rows),
        success_rate=sum(r["success"] for r in rows) / len(rows) if rows else None,
        score_mean=statistics.mean(r["score"] for r in rows) if rows else None,
        episodes=[{k: r.get(k) for k in ("task_name", "success", "score", "episode_step", "reason")} for r in rows],
    )
    write_json(run / "summary.json", value)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execution", choices=["legacy", "optimized-eager", "compile", "compile-graph"], default="compile")
    p.add_argument("--port", type=int, default=8017)
    p.add_argument("--server", action="store_true")
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.server:
        server(args)
        return
    for path in (PYTHON, ROBOLAB / ".venv/bin/python", VAE, ROBOLAB / "Cosmos3-Edge-Policy-DROID/model_index.json"):
        if not path.exists():
            raise FileNotFoundError(path)
    with socket.socket() as check:
        check.bind(("127.0.0.1", args.port))
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [
        "cosmos_framework/inference/edge_core_stable.py",
        "cosmos_framework/inference/edge_core_stable_fast.py",
        "cosmos_framework/model/generator/mot/unified_mot.py",
        "cosmos_framework/scripts/action_policy_server_robolab_version1.py",
        "tools/run_asi_optimized_smoke10.py",
    ]
    write_json(
        args.output / "manifest.json",
        dict(
            tasks=TASKS,
            execution=args.execution,
            seed=0,
            simulator_seed=0,
            reset_policy_rng_each_task=True,
            deterministic_seed=False,
            steps=4,
            shift=5,
            guidance=3,
            core80=True,
            stable104=True,
            top_blocks=6,
            retries=0,
            video_mode="none",
            checkpoint=str(ROBOLAB / "Cosmos3-Edge-Policy-DROID"),
            vae=str(VAE),
            head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
            timing_note="Observed generation-wrapper latency with Isaac active; not isolated warm chunk speedup",
        ),
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "LD_LIBRARY_PATH": "",
        "CUDA_VISIBLE_DEVICES": "0",
        "COSMOS_TRAINING": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "OMNI_KIT_ACCEPT_EULA": "Y",
        "ACCEPT_EULA": "Y",
        "PRIVACY_CONSENT": "Y",
    }
    server_cmd = [
        str(PYTHON),
        "-m",
        "tools.run_asi_optimized_smoke10",
        "--server",
        "--execution",
        args.execution,
        "--output",
        str(args.output),
        "--port",
        str(args.port),
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
        str(args.output / "simulator"),
    ]
    write_json(args.output / "commands.json", dict(server=server_cmd, simulator=sim_cmd))
    worker = sim = None
    try:
        aggregate(args.output, "starting")
        with (args.output / "server.log").open("w") as log:
            worker = subprocess.Popen(
                server_cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        deadline = time.monotonic() + 600
        while "OPTIMIZED_SMOKE_READY" not in (args.output / "server.log").read_text(errors="replace"):
            if worker.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Server startup failed")
            time.sleep(2)
        print(f"[ASI {args.execution}] server ready; 10 tasks, simulator seed0 / policy seed0", flush=True)
        with (args.output / "simulator.log").open("w") as log:
            sim = subprocess.Popen(
                sim_cmd,
                cwd=ROBOLAB,
                env={**env, "PYTHONPATH": f"{ROBOLAB}:/root/robolab/cosmos-edge-overlay"},
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        last = -1
        while sim.poll() is None:
            if worker.poll() is not None:
                raise RuntimeError("Policy server exited during evaluation")
            result = aggregate(args.output, "running")
            if result["completed"] != last:
                last = result["completed"]
                score = result["score_mean"]
                print(f"[ASI] {last}/10 completed | success {result['successes']}/{last} | score {score}", flush=True)
                if last:
                    print(json.dumps(result["episodes"][-1], ensure_ascii=False), flush=True)
            time.sleep(5)
        result = aggregate(args.output, "complete" if sim.returncode == 0 else "failed")
        if result["completed"] != 10 or sim.returncode != 0:
            raise RuntimeError(f"Evaluation incomplete: {result['completed']}/10, returncode={sim.returncode}")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except BaseException:
        aggregate(args.output, "stopped_error")
        raise
    finally:
        stop_owned(sim)
        stop_owned(worker)


if __name__ == "__main__":
    main()
