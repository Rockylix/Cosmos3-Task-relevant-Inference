"""Small-record scan server; only Dense stores temporary paired inputs."""

from cosmos_framework.inference.common.init import init_script

init_script()

import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.inference.toca_future import ToCaFutureConfig, ToCaFutureController
from cosmos_framework.inference.toca_scan_plan import ROBOLAB, TASKS
from cosmos_framework.scripts.action_policy_server_comparison import cpu_tree
from cosmos_framework.scripts.action_policy_server_robolab import (
    RobolabServerArgs,
    _load_openpi_websocket_policy_server,
)
from cosmos_framework.scripts.paired_future_fidelity import EagerService, write_json
from cosmos_framework.utils import log


class ScanArgs(RobolabServerArgs):
    format_prompt_as_json: bool | None = True
    scan_config: Path
    scan_output: Path
    scan_capture_root: Path | None = None


class ScanService(EagerService):
    def __init__(self, args):
        if (args.num_steps, args.guidance, args.shift, args.format_prompt_as_json) != (4, 3, 5, True):
            raise ValueError("Scan requires 4 steps, guidance=3, shift=5, structured prompt")
        settings = json.loads(args.scan_config.read_text())
        self.scan_mode = settings["mode"]
        values = settings["config"]
        self.scan_config = None
        if values is not None:
            self.scan_config = ToCaFutureConfig(**{**values, "full_steps": tuple(values["full_steps"])})
        self.scan_output = args.scan_output.resolve()
        self.scan_output.mkdir(parents=True, exist_ok=True)
        if (self.scan_output / "requests.jsonl").exists():
            raise FileExistsError("Each server attempt needs its own output directory")
        metadata = json.loads((ROBOLAB / "robolab/tasks/_metadata/task_metadata.json").read_text())
        self.task_for_prompt = {d["instruction"]: d["task_name"] for d in metadata if d["task_name"] in TASKS}
        self.scan_prompt, self.scan_chunk, self.scan_request = None, 0, 0
        if args.scan_capture_root is not None:
            if self.scan_mode != "dense":
                raise ValueError("Only Dense captures temporary inputs")
            args.scan_capture_root.mkdir(parents=True, exist_ok=True)
        super().__init__(args)
        original = self.model.generate_samples_from_batch
        write_json(
            self.scan_output / "runtime.json",
            {
                **settings,
                "compile": False,
                "cuda_graphs": False,
                "seed": args.seed,
                "reset_rng_each_task": True,
                "deterministic_seed": args.deterministic_seed,
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
            },
        )

        def generate(*positional, **kwargs):
            capture = args.scan_capture_root is not None and self.scan_chunk <= 3
            paired = cpu_tree({"args": positional, "kwargs": kwargs}) if capture else None
            torch.cuda.synchronize()
            start = time.perf_counter()
            controller = (
                ToCaFutureController(self.model.net, self.scan_config) if self.scan_config is not None else None
            )
            with controller if controller is not None else nullcontext():
                samples = original(*positional, **kwargs)
            summary = controller.finish() if controller is not None else {}
            for key in ("action", "vision"):
                if not torch.isfinite(samples[key][0]).all().item():
                    raise FloatingPointError(f"Nonfinite {key}")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            row = {
                "task": self.task_for_prompt[self.scan_prompt],
                "chunk": self.scan_chunk,
                "seed": kwargs["seed"],
                "generation_s": elapsed,
                "all_finite": True,
            }
            with (self.scan_output / "requests.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if self.scan_request == 0 and summary:
                write_json(self.scan_output / "compute_check.json", {k: v for k, v in summary.items() if k != "layout"})
            if capture:
                path = args.scan_capture_root / f"{row['task']}.pt"
                torch.save(
                    {
                        **paired,
                        "metadata": {**row, "prompt_chunk": self.scan_chunk, "prompt": self.scan_prompt},
                        "outputs": {k: samples[k][0].detach().cpu() for k in ("action", "vision")},
                    },
                    path,
                )
            self.scan_request += 1
            return samples

        self.model.generate_samples_from_batch = generate

    def infer(self, obs):
        prompt = obs["prompt"]
        if prompt not in self.task_for_prompt:
            raise ValueError(f"Unexpected scan task prompt: {prompt}")
        if prompt != self.scan_prompt:
            self.scan_prompt, self.scan_chunk = prompt, 0
            self._rng = np.random.default_rng(self.cfg.seed)
        self.scan_chunk += 1
        return super().infer(obs)


def main():
    from cosmos_framework.inference.common.args import tyro_cli

    args = tyro_cli(ScanArgs, description=__doc__)
    service = ScanService(args)
    log.info(f"[scan-server] READY {args.host}:{args.port} {service.scan_mode}")
    _load_openpi_websocket_policy_server()(
        policy=service, host=args.host, port=args.port, metadata={"strategy": service.scan_mode}
    ).serve_forever()


if __name__ == "__main__":
    main()
