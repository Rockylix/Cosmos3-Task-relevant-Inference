"""Move an in-flight ASI evaluation supervisor to tmux without restarting GPU work."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from run_asi_smoke10 import ROOT, aggregate, read_episodes, stop_owned, write_json

LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
LIBC.syscall.restype = ctypes.c_long


def pidfd_call(name, *args):
    # Linux x86_64: support portable Python builds / glibc < 2.36 without wrappers.
    if os.uname().machine != "x86_64":
        raise RuntimeError("This handoff utility supports this host's x86_64 ABI only")
    number = {"pidfd_open": 434, "pidfd_send_signal": 424}[name]
    result = LIBC.syscall(ctypes.c_long(number), *args)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


class ExistingProcess:
    """Pin process identity with pidfd; never signal a reused PID."""

    def __init__(self, pid, required_args):
        self.pid = pid
        self.fd = pidfd_call("pidfd_open", pid, 0)
        self.command = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        if not all(arg in self.command for arg in required_args):
            raise RuntimeError(f"PID {pid} is not the expected evaluation process")

    def poll(self):
        import select

        return 0 if select.select([self.fd], [], [], 0)[0] else None

    def send(self, sig):
        pidfd_call("pidfd_send_signal", self.fd, int(sig), None, 0)

    def wait(self, timeout):
        import select

        if not select.select([self.fd], [], [], timeout)[0]:
            raise subprocess.TimeoutExpired(str(self.pid), timeout)
        return 0

    def stop(self):
        if self.poll() is None:
            self.send(signal.SIGTERM)
            try:
                self.wait(20)
            except subprocess.TimeoutExpired:
                self.send(signal.SIGKILL)
                self.wait(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--driver-pid", type=int, required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--simulator-pid", type=int, required=True)
    args = parser.parse_args()
    run = args.output.resolve()
    driver = ExistingProcess(args.driver_pid, ["tools/run_asi_smoke10.py", str(run)])
    server = ExistingProcess(args.server_pid, ["cosmos_framework.scripts.asi_smoke", str(run), "serve"])
    simulator = ExistingProcess(args.simulator_pid, ["policies/cosmos3/run.py", str(run / "simulator")])
    commands = json.loads((run / "commands.json").read_text())
    env = {
        pair.split("=", 1)[0]: pair.split("=", 1)[1]
        for pair in Path(f"/proc/{server.pid}/environ").read_bytes().decode().split("\0")
        if "=" in pair
    }
    env["TMUX"] = os.environ.get("TMUX", "")
    env["TMUX_PANE"] = os.environ.get("TMUX_PANE", "")
    if (run / "tmux_handoff.json").exists():
        raise RuntimeError("An existing handoff must not be overwritten")

    def log(message):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with (run / "tmux_progress.log").open("a") as stream:
            stream.write(line + "\n")

    # Pause ONLY the old Python supervisor. GPU children have independent sessions.
    driver.send(signal.SIGSTOP)
    try:
        if any(p.poll() is not None for p in (server, simulator)):
            raise RuntimeError("GPU stage changed during handoff; old supervisor will resume")
        if json.loads((run / "summary.json").read_text())["phase"] != "closed_loop":
            raise RuntimeError("Only a live closed-loop phase can be transferred")
        for process in (server, simulator):
            fields = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[1]) != driver.pid or os.getsid(process.pid) != process.pid:
                raise RuntimeError("Unexpected child ownership/session; refuse handoff")
        write_json(
            run / "tmux_handoff.json",
            {
                "supervisor_pid_before": driver.pid,
                "supervisor_pid_after": os.getpid(),
                "server_pid_unchanged": server.pid,
                "simulator_pid_unchanged": simulator.pid,
                "tmux": os.environ.get("TMUX"),
                "episodes_preserved": len(read_episodes(run)),
                "protocol_changed": False,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
    except BaseException:
        driver.send(signal.SIGCONT)
        raise
    # Do not invoke the old supervisor's finally block, which would stop its children.
    driver.send(signal.SIGKILL)
    driver.wait(10)
    benchmark = None
    try:
        log("ASI tmux supervision active; server/simulator unchanged; no episodes restarted")
        previous = None
        while simulator.poll() is None:
            if server.poll() is not None:
                raise RuntimeError("Server exited during closed-loop evaluation")
            val = aggregate(run, "closed_loop")
            signature = val["completed"]
            if signature != previous:
                previous = signature
                score = "—" if val["score_mean"] is None else f"{val['score_mean']:.4f}"
                log(f"ASI {signature}/10 completed | success {val['success_count']}/{signature} | score {score}")
            time.sleep(5)
        if len(read_episodes(run)) != 10:
            raise RuntimeError("Simulation exited without 10 distinct results; no automatic retries")
        server.stop()
        aggregate(run, "benchmark")
        log("Closed-loop complete; paired Dense/ASI timing and future-frame fidelity start")
        with (run / "benchmark.log").open("x") as stream:
            benchmark = subprocess.Popen(
                commands["benchmark"],
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        previous = None
        while benchmark.poll() is None:
            val = aggregate(run, "benchmark")
            if val["paired_tasks"] != previous:
                previous = val["paired_tasks"]
                log(f"Paired timing / future fidelity: {previous}/10 tasks")
            time.sleep(5)
        if benchmark.returncode != 0:
            raise RuntimeError("Benchmark failed; preserve closed-loop results and inputs")
        val = aggregate(run, "complete")
        if val["paired_tasks"] != 10:
            raise RuntimeError("Benchmark exited without all 10 paired metrics")
        log(json.dumps(val, ensure_ascii=False, indent=2))
    except BaseException:
        aggregate(run, "stopped_error")
        raise
    finally:
        simulator.stop()
        server.stop()
        stop_owned(benchmark)


if __name__ == "__main__":
    main()
