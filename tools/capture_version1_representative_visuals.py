#!/usr/bin/env python3
"""Capture one Version1 request for representative simple/moderate/complex tasks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tools.run_bibd_6seeds_6tasks_3strategies import (
    EDGE_ROOT,
    EXPERIMENT_ROOT,
    ROBOLAB_ROOT,
    make_environment,
    policy_server_command,
    simulator_command,
    stop_process_group,
    wait_for_server,
)

TASKS = (
    ("simple", "BananaInBowlTask"),
    ("moderate", "RubiksCubeLeftOfBowlTask"),
    ("complex", "RubiksCubesInBinTask"),
)
OUTPUT_NAME = "bibd6x6_v6_visual_capture_v1"


def main() -> None:
    visual_root = EXPERIMENT_ROOT / "visual_samples"
    server_dir = visual_root / "server"
    log_dir = visual_root / "logs"
    simulator_output = ROBOLAB_ROOT / "output" / OUTPUT_NAME
    if visual_root.exists() or simulator_output.exists():
        raise FileExistsError(f"Use fresh capture paths: {visual_root}, {simulator_output}")
    server_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    server_log = log_dir / "server.log"
    simulator_log = log_dir / "simulator.log"
    env = make_environment()

    server_command = policy_server_command("direct_core64", 579362556, server_dir)
    server_command.append("--capture-first-mask-overlay-per-prompt")
    task_names = [task for _, task in TASKS]
    sim_command = simulator_command(task_names, OUTPUT_NAME)
    cap_index = sim_command.index("--max-episode-steps") + 1
    sim_command[cap_index] = "1"

    with server_log.open("w", encoding="utf-8") as stream:
        server = subprocess.Popen(
            server_command,
            cwd=EDGE_ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        wait_for_server(server, server_log)
        with simulator_log.open("w", encoding="utf-8") as stream:
            simulator = subprocess.run(
                sim_command,
                cwd=ROBOLAB_ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if simulator.returncode != 0:
            tail = "\n".join(simulator_log.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
            raise RuntimeError(f"visual simulator exited with code {simulator.returncode}\n{tail}")
    finally:
        stop_process_group(server, "visual policy server")

    request_dirs = sorted(server_dir.glob("request_*"))
    if len(request_dirs) != len(TASKS):
        raise RuntimeError(f"Expected {len(TASKS)} request captures, found {len(request_dirs)}")
    task_results = {
        str(item["task_name"]): item
        for item in (
            json.loads(line)
            for line in (simulator_output / "episode_results.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    task_by_prompt = {str(item["instruction"]): task for task, item in task_results.items()}
    difficulty_by_task = {task: difficulty for difficulty, task in TASKS}
    index = []
    for request_dir in request_dirs:
        capture = json.loads((request_dir / "mask_overlay_capture.json").read_text(encoding="utf-8"))
        prompt = str(capture["prompt"])
        if prompt not in task_by_prompt:
            raise RuntimeError(f"Cannot map captured prompt to a task: {prompt!r}")
        task = task_by_prompt[prompt]
        difficulty = difficulty_by_task[task]
        required = (
            request_dir / "step0_v5_2_motion_core_stable_adaptive.pt",
            request_dir / "predicted_future_frames" / "frame_004.png",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"Capture {request_dir} is incomplete: {missing}")
        index.append(
            {
                "difficulty": difficulty,
                "task": task,
                "prompt": prompt,
                "request_dir": str(request_dir),
                "episode_success": bool(task_results[task]["success"]),
                "episode_step": int(task_results[task]["episode_step"]),
                "capture_cap_steps": 1,
            }
        )
    (visual_root / "capture_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(index, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
