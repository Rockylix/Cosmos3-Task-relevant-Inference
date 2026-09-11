"""Durable serial 25-config/10-task scan; never retry completed episodes."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import signal
import socket
import statistics
import subprocess
import time
from pathlib import Path

from cosmos_framework.inference.toca_scan_plan import DIFFICULTY, EDGE_PYTHON, ROBOLAB, ROOT, TASKS, VAE, configurations


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_episodes(path):
    if not path.exists():
        return []
    rows = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    names = [r["task_name"] for r in rows]
    if len(names) != len(set(names)) or not set(names) <= set(TASKS):
        raise RuntimeError("Duplicate/unexpected episodes; refusing to filter outcomes")
    if any(not isinstance(r.get("success"), bool) or r.get("score") is None for r in rows):
        raise RuntimeError("Episode result lacks success/score")
    return rows


def stop_owned(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def environment():
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
    return env


def progress(run, **fields):
    write_json(run / "progress.json", {"updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields})


def aggregate(run):
    paired = [json.loads(p.read_text()) for task in TASKS if (p := run / "paired" / f"{task}.json").exists()]
    rows = []
    for mode, config in configurations().items():
        eps = read_episodes(run / mode / "simulator/episode_results.jsonl")
        row = {
            "mode": mode,
            "schedule": "DDDD" if config is None else ("DCDC" if len(config["full_steps"]) == 2 else "DCCC"),
            "r": None if config is None else config["fresh_ratio"],
            "bonus": 0 if config is None else config["spatial_bonus"],
            "cfg": "native" if config is None else config["cfg_selection"],
            "completed": len(eps),
            "expected": 10,
            "success_count": sum(r["success"] for r in eps),
            "success_rate": sum(r["success"] for r in eps) / 10 if len(eps) == 10 else None,
            "score_mean": statistics.mean(r["score"] for r in eps) if len(eps) == 10 else None,
            "simple_success": sum(r["success"] for r in eps if DIFFICULTY[r["task_name"]] == "simple"),
            "moderate_success": sum(r["success"] for r in eps if DIFFICULTY[r["task_name"]] == "moderate"),
            "paired_tasks": len(paired),
            "chunk_mean_s": None,
            "chunk_median_macro_s": None,
            "chunk_p90_macro_s": None,
            "speedup": None,
            "rgb_cosine": None,
            "rgb_relative_l2": None,
            "psnr_db": None,
            "ssim": None,
        }
        if len(paired) == 10:
            for source, dest in (
                ("mean_s", "chunk_mean_s"),
                ("median_s", "chunk_median_macro_s"),
                ("p90_s", "chunk_p90_macro_s"),
            ):
                row[dest] = statistics.mean(p["timing"][mode][source] for p in paired)
            base = statistics.mean(p["timing"]["dense"]["median_s"] for p in paired)
            row["speedup"] = base / row["chunk_median_macro_s"]
            for key in ("rgb_cosine", "rgb_relative_l2", "psnr_db", "ssim"):
                vals = [p["fidelity"][mode][key] for p in paired]
                row[key] = statistics.mean(vals) if all(x is not None for x in vals) else None
        rows.append(row)
    temp = run / "summary.csv.tmp"
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(run / "summary.csv")

    def value(v, digits=4):
        return "—" if v is None else f"{v:.{digits}f}"

    report = [
        "# ToCa 24配置 × 10任务扫描",
        "",
        "任务：8 simple + 2 moderate；每配置10 episodes。空白表示未完成，不是零分。",
        "",
        "计时为离线同输入、同seed，预热5次后轮换30轮；时间列为10个任务各自统计量的等权平均，非混合请求的全局中位数。",
        "RGB来自同一Dense轨迹输入，固定[-1,1]→[0,1]映射，比较future 32帧；PSNR/SSIM先逐帧再逐任务平均。Dense PSNR为∞。",
        "",
        "| 配置 | 完成 | 成功 | Score | Chunk median(s) | Speedup | RGB cos | RGB rel-L2 | PSNR | SSIM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        report.append(
            f"| {r['mode']} | {r['completed']}/10 | {r['success_count']}/{r['completed']} | {value(r['score_mean'])} | {value(r['chunk_median_macro_s'], 6)} | {value(r['speedup'], 3)} | {value(r['rgb_cosine'], 6)} | {value(r['rgb_relative_l2'], 6)} | {value(r['psnr_db'], 3)} | {value(r['ssim'], 6)} |"
        )
    report.extend(["", "本表是固定seed的超参筛选结果，不声明未见任务或多seed泛化。闭环失败不重试、不筛选。", ""])
    (run / "report_cn.md").write_text("\n".join(report))
    return rows


def cleanup_episode_raw(run, mode):
    removed = []
    for task in TASKS:
        # Only files produced by this scan; result summaries are already durable.
        for filename in ("run_0.hdf5", "log_0_env0.json"):
            path = run / mode / "simulator" / task / filename
            if path.is_file() and not path.is_symlink():
                removed.append({"path": str(path.relative_to(run)), "bytes": path.stat().st_size})
                path.unlink()
    if removed:
        with (run / "cleanup.jsonl").open("a") as stream:
            stream.write(json.dumps({"mode": mode, "removed": removed}) + "\n")


def run_mode(run, mode, config, port, env):
    mode_dir = run / mode
    mode_dir.mkdir(exist_ok=True)
    ep_file = mode_dir / "simulator/episode_results.jsonl"
    if len(read_episodes(ep_file)) == 10:
        print(f"[skip] {mode}: all ten episodes already completed", flush=True)
        return
    attempt = mode_dir / f"attempt_{time.time_ns()}"
    attempt.mkdir()
    config_file = mode_dir / "config.json"
    write_json(config_file, {"mode": mode, "config": config})
    server_cmd = [
        str(EDGE_PYTHON),
        "-u",
        "-m",
        "cosmos_framework.scripts.action_policy_server_toca_scan",
        "--scan-config",
        str(config_file),
        "--scan-output",
        str(attempt / "server"),
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
        str(attempt / "model_output"),
        "--experiment-overrides",
        f"model.config.tokenizer.vae_path={VAE}",
        "model.config.tokenizer.object_store_credential_path_pretrained=",
        "model.config.tokenizer.bucket_name=",
    ]
    if mode == "dense":
        server_cmd += ["--scan-capture-root", str(run / "_temporary_inputs")]
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
        str(mode_dir / "simulator"),
    ]
    write_json(attempt / "commands.json", {"server": server_cmd, "simulator": sim_cmd})
    with socket.socket() as check:
        check.bind(("127.0.0.1", port))
    server = simulator = None
    try:
        with (attempt / "server.log").open("w") as server_log, (attempt / "simulator.log").open("w") as sim_log:
            server = subprocess.Popen(
                server_cmd, cwd=ROOT, env=env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True
            )
            progress(run, phase="server_start", mode=mode, server_pid=server.pid, attempt=str(attempt))
            for _ in range(300):
                if server.poll() is not None:
                    raise RuntimeError(f"{mode} server exited; see {attempt}/server.log")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    time.sleep(2)
            else:
                raise RuntimeError("Server readiness timeout")
            sim_env = {**env, "PYTHONPATH": f"{ROBOLAB}:/root/robolab/cosmos-edge-overlay"}
            simulator = subprocess.Popen(
                sim_cmd, cwd=ROBOLAB, env=sim_env, stdout=sim_log, stderr=subprocess.STDOUT, start_new_session=True
            )
            previous = -1
            while simulator.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError(f"{mode} server stopped during simulation")
                eps = read_episodes(ep_file)
                if len(eps) != previous:
                    previous = len(eps)
                    rows = aggregate(run)
                    total = sum(r["completed"] for r in rows)
                    succ = sum(r["success"] for r in eps)
                    score = statistics.mean(r["score"] for r in eps) if eps else 0
                    print(
                        f"[closed-loop] {mode} completed={len(eps)}/10 success={succ}/{len(eps)} score={score:.4f} total={total}/250",
                        flush=True,
                    )
                    progress(
                        run,
                        phase="closed_loop",
                        mode=mode,
                        completed=len(eps),
                        total_episodes=total,
                        server_pid=server.pid,
                        simulator_pid=simulator.pid,
                        attempt=str(attempt),
                    )
                time.sleep(2)
            eps = read_episodes(ep_file)
            if simulator.returncode != 0 or len(eps) != 10:
                raise RuntimeError(
                    f"{mode} incomplete, exit={simulator.returncode}, episodes={len(eps)}; no automatic retry"
                )
    finally:
        stop_owned(simulator)
        stop_owned(server)
        aggregate(run)
    cleanup_episode_raw(run, mode)
    print(
        f"[completed] {mode}: {sum(r['success'] for r in eps)}/10 score={statistics.mean(r['score'] for r in eps):.4f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8019)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if run.exists() and not args.resume:
        raise FileExistsError("Use --resume for the existing exact same scan; never overwrite")
    if ROOT / "experiments" not in run.parents:
        raise ValueError("Scan output must be under the ToCa experiment worktree")
    run.mkdir(parents=True, exist_ok=True)
    with (run / "scan.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gate = json.loads((run / "validation/summary.json").read_text())
        checks = gate.get("checks", [])
        if (
            gate.get("status") != "complete"
            or not gate.get("default_joint_regression_exact")
            or [c["mode"] for c in checks] != list(configurations())
            or not all(c.get("actual_module_rows_pass") and c.get("finite") for c in checks)
        ):
            raise RuntimeError("All 25 GPU validation cases must pass before starting episodes")
        env = environment()
        for path in (EDGE_PYTHON, ROBOLAB / ".venv/bin/python", VAE, ROBOLAB / "Cosmos3-Edge-Policy-DROID"):
            if not path.exists():
                raise FileNotFoundError(path)
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip()
        if active:
            raise RuntimeError(f"GPU occupied by existing processes {active}; not stopping them")
        meta = {
            d["task_name"]: d for d in json.loads((ROBOLAB / "robolab/tasks/_metadata/task_metadata.json").read_text())
        }
        for task in TASKS:
            assert meta[task]["difficulty_label"] == DIFFICULTY[task]
        sources = [
            "cosmos_framework/inference/toca_future.py",
            "cosmos_framework/inference/toca_joint_attention.py",
            "cosmos_framework/inference/toca_scan_plan.py",
            "cosmos_framework/scripts/action_policy_server_toca_scan.py",
            "cosmos_framework/scripts/benchmark_toca_scan.py",
            "tools/run_toca_scan.py",
        ]
        manifest = {
            "tasks": TASKS,
            "difficulty": DIFFICULTY,
            "configurations": configurations(),
            "episodes": 250,
            "simulator_seed": 0,
            "policy_seed": 0,
            "deterministic_seed": False,
            "reset_rng_each_episode": True,
            "steps": 4,
            "shift": 5,
            "guidance": 3,
            "compile": False,
            "cuda_graphs": False,
            "step_limits": "official per-task",
            "videos": False,
            "paired_warmups": 5,
            "paired_repeats": 30,
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        }
        if (run / "manifest.json").exists():
            if json.loads((run / "manifest.json").read_text()) != manifest:
                raise RuntimeError("Manifest/code changed; refusing to mix runs")
        else:
            write_json(run / "manifest.json", manifest)
        configs = configurations()
        try:
            run_mode(run, "dense", None, args.port, env)
            if not (run / "paired/summary.json").exists():
                for task in TASKS:
                    if not (run / "_temporary_inputs" / f"{task}.pt").exists():
                        raise RuntimeError(f"Missing paired input for {task}")
                command = [
                    str(EDGE_PYTHON),
                    "-u",
                    "-m",
                    "cosmos_framework.scripts.benchmark_toca_scan",
                    "--run-dir",
                    str(run),
                ]
                print("[paired] start 10 inputs x 25 modes; warmed timing and RGB errors, no saved images", flush=True)
                worker = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                progress(run, phase="paired", worker_pid=worker.pid)
                try:
                    with (run / "paired.log").open("a") as log:
                        for line in worker.stdout:
                            log.write(line)
                            if "[paired]" in line or "Traceback" in line or "Error" in line:
                                print(line.rstrip(), flush=True)
                                aggregate(run)
                    if worker.wait() != 0:
                        raise RuntimeError("Paired benchmark failed; see paired.log")
                finally:
                    stop_owned(worker)
            aggregate(run)
            removed = []
            for task in TASKS:
                path = run / "_temporary_inputs" / f"{task}.pt"
                if path.is_file() and not path.is_symlink():
                    removed.append({"path": str(path.relative_to(run)), "bytes": path.stat().st_size})
                    path.unlink()
            if removed:
                with (run / "cleanup.jsonl").open("a") as stream:
                    stream.write(json.dumps({"paired_inputs_removed": removed}) + "\n")
                print("[cleanup] temporary Dense inputs removed after paired metrics were saved", flush=True)
            for mode, config in list(configs.items())[1:]:
                run_mode(run, mode, config, args.port, env)
            rows = aggregate(run)
            assert sum(r["completed"] for r in rows) == 250
            progress(run, phase="complete", total_episodes=250)
            print(f"[DONE] 250/250 episodes; final table: {run}/summary.csv", flush=True)
        except BaseException as exc:
            progress(run, phase="stopped_error", error=repr(exc))
            aggregate(run)
            raise


if __name__ == "__main__":
    main()
