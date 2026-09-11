"""WorldCache GPU gate, ten-task server, and paired eager timing. No simulation videos."""

from cosmos_framework.inference.common.init import init_script

init_script()

import copy
import json
import random
import statistics
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from cosmos_framework.scripts.action_policy_server_robolab import (
    RobolabPolicyService,
    RobolabServerArgs,
    _load_openpi_websocket_policy_server,
)
from cosmos_framework.utils import log


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def metrics(test, reference):
    if test.shape != reference.shape:
        raise ValueError("Metric input shapes differ")
    # Long RGB vectors require FP64 reductions; FP32 norm reduction can yield cos > 1.
    x, y = test.double().reshape(-1), reference.double().reshape(-1)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise FloatingPointError("Nonfinite comparison output")
    cosine = float(torch.dot(x, y) / (x.norm() * y.norm()).clamp_min(1e-24))
    if not -1 - 1e-12 <= cosine <= 1 + 1e-12:
        raise ArithmeticError(f"Invalid FP64 cosine {cosine}")
    return {
        "mse": float((x - y).square().mean()),
        "cosine": max(-1.0, min(1.0, cosine)),
        "relative_l2": float((x - y).norm() / y.norm().clamp_min(1e-12)),
        "max_absolute_error": float((x - y).abs().max()),
    }


class Args(RobolabServerArgs):
    worldcache_dddc: bool = True
    eager: bool = True
    format_prompt_as_json: bool | None = True
    eval_root: Path
    eval_phase: Literal["serve", "benchmark", "capture"] = "serve"
    gate_capture: Path


class Service(RobolabPolicyService):
    def __init__(self, args):
        if (args.num_steps, args.shift, args.guidance, args.seed, args.deterministic_seed) != (4, 5, 3, 0, False):
            raise ValueError("Frozen scan protocol: steps4 shift5 guidance3 policyseed0 deterministic_seed=False")
        if not (args.worldcache_dddc and args.eager and args.format_prompt_as_json):
            raise ValueError("Expected native eager WorldCache and structured prompt")
        self.run = args.eval_root.resolve()
        self.manifest = json.loads((self.run / "manifest.json").read_text())
        metadata = json.loads(Path("/root/robolab/RoboLab/robolab/tasks/_metadata/task_metadata.json").read_text())
        self.task_for_prompt = {
            row["instruction"]: row["task_name"] for row in metadata if row["task_name"] in self.manifest["tasks"]
        }
        self.prompt, self.chunk = None, 0
        super().__init__(args)
        self.original_generate = self.model.generate_samples_from_batch
        write_json(
            self.run / f"runtime_{args.eval_phase}.json",
            {
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "compile": False,
                "cuda_graphs": False,
                "source": str(Path(__file__).resolve()),
                "seed": 0,
                "deterministic_seed": False,
                "reset_rng_each_task": True,
            },
        )
        if args.eval_phase != "benchmark":
            if args.eval_phase == "serve":
                self.gate(torch.load(args.gate_capture, weights_only=False, map_location="cpu"))
            if (self.run / "requests.jsonl").exists():
                raise FileExistsError("Refusing to overwrite existing closed-loop requests")
            self.model.generate_samples_from_batch = self.generate_logged

    @torch.inference_mode()
    def run_paired(self, capture, sparse, audit=False):
        positional, kwargs = copy.deepcopy(capture["args"]), copy.deepcopy(capture["kwargs"])
        kwargs["worldcache_config"] = self._worldcache_config if sparse else None
        seed = kwargs["seed"][0]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        trace = []
        hook = self.model.net.register_forward_hook(lambda m, a, out: trace.append(cpu_tree(out))) if audit else None
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            samples = self.original_generate(*positional, **kwargs)
        finally:
            if hook:
                hook.remove()
        for key in ("action", "vision"):
            if not torch.isfinite(samples[key][0]).all().item():
                raise FloatingPointError(f"Nonfinite {key}")
        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        output = {key: cpu_tree(samples[key][0]) for key in ("action", "vision")}
        return output, duration, trace

    def gate(self, capture):
        dense, _, dense_trace = self.run_paired(capture, False, audit=True)
        sparse, _, sparse_trace = self.run_paired(capture, True, audit=True)
        if (len(dense_trace), len(sparse_trace)) != (8, 6):
            raise AssertionError(f"Incorrect true forward counts {len(dense_trace)}, {len(sparse_trace)}")
        for d, s in zip(dense_trace[:6], sparse_trace, strict=True):
            for key in ("preds_vision", "preds_action"):
                torch.testing.assert_close(d[key][0], s[key][0], rtol=0, atol=0)
        reference = {k: metrics(dense[k], capture["outputs"][k]) for k in ("action", "vision")}
        report = {
            "passed": True,
            "actual_dense_forwards": 8,
            "actual_worldcache_forwards": 6,
            "first_three_steps_both_branches_bitwise_equal": True,
            "all_finite": True,
            "native_latent_shape": list(dense["vision"].shape),
            "comparison_to_stored_dense": reference,
            "worldcache_vs_dense": {k: metrics(sparse[k], dense[k]) for k in ("action", "vision")},
            "controller": self.model._last_worldcache_report,
            "note": "Gate timings include audit copies and MUST NOT be used as performance measurements",
        }
        write_json(self.run / "gpu_gate.json", report)
        torch.cuda.empty_cache()
        log.info("[worldcache] GPU gate PASS: Dense8 -> WorldCache6, first three steps bitwise equal")

    def infer(self, obs):
        prompt = obs["prompt"]
        if prompt not in self.task_for_prompt:
            raise ValueError(f"Unexpected task prompt {prompt}")
        if prompt != self.prompt:
            self.prompt, self.chunk = prompt, 0
            self._rng = np.random.default_rng(self.cfg.seed)
        self.chunk += 1
        return super().infer(obs)

    def generate_logged(self, *positional, **kwargs):
        captured = cpu_tree({"args": positional, "kwargs": kwargs}) if self.chunk == 3 else None
        torch.cuda.synchronize()
        start = time.perf_counter()
        samples = self.original_generate(*positional, **kwargs)
        for key in ("action", "vision"):
            if not torch.isfinite(samples[key][0]).all().item():
                raise FloatingPointError(f"Nonfinite {key}")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        report = self.model._last_worldcache_report
        if (report["full_forwards"], report["cache_forwards"]) != (6, 2):
            raise AssertionError("Bad WorldCache controller counts")
        row = {
            "task": self.task_for_prompt[self.prompt],
            "chunk": self.chunk,
            "seed": kwargs["seed"],
            "generation_s": elapsed,
            "all_finite": True,
            "full_forwards": 6,
            "cache_forwards": 2,
        }
        with (self.run / "requests.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if captured is not None:
            folder = self.run / "_temporary_inputs"
            folder.mkdir(exist_ok=True)
            torch.save({**captured, "metadata": row}, folder / f"{row['task']}.pt")
        return samples

    @torch.inference_mode()
    def benchmark(self):
        for task in self.manifest["tasks"]:
            path = self.run / "_temporary_inputs" / f"{task}.pt"
            capture = torch.load(path, weights_only=False, map_location="cpu")
            timings = {"dense": [], "worldcache": []}
            for i in range(5):
                for sparse in (False, True) if i % 2 == 0 else (True, False):
                    self.run_paired(capture, sparse)
            outputs = {}
            for i in range(30):
                for sparse in (False, True) if i % 2 == 0 else (True, False):
                    key = "worldcache" if sparse else "dense"
                    output, duration, _ = self.run_paired(capture, sparse)
                    timings[key].append(duration)
                    outputs[key] = output
            rgb = {}
            for key, output in outputs.items():
                decoded = self.model.decode(output["vision"].cuda()).float().cpu()
                # Native decode [1,3,33,H,W]; exclude the conditioned first image.
                if decoded.ndim != 5 or decoded.shape[1:3] != (3, 33):
                    raise ValueError(f"Unexpected decoded shape {decoded.shape}")
                rgb[key] = (decoded[:, :, 1:].clamp(-1, 1) + 1) / 2
            stats = {
                key: {
                    "mean_s": statistics.mean(v),
                    "median_s": statistics.median(v),
                    "p90_s": float(np.quantile(v, 0.9)),
                    "samples_s": v,
                }
                for key, v in timings.items()
            }
            result = {
                "task": task,
                "capture": capture["metadata"],
                "warmups_each": 5,
                "repeats_each": 30,
                "input_source": "WorldCache closed-loop chunk3; same inputs/noise for Dense and WorldCache",
                "timing": stats,
                "speedup": stats["dense"]["median_s"] / stats["worldcache"]["median_s"],
                "action": metrics(outputs["worldcache"]["action"][1:], outputs["dense"]["action"][1:]),
                "future_latent": metrics(
                    outputs["worldcache"]["vision"][:, :, 1:], outputs["dense"]["vision"][:, :, 1:]
                ),
                "future_rgb": metrics(rgb["worldcache"], rgb["dense"]),
                "fidelity_precision": "fp64",
            }
            write_json(self.run / "paired" / f"{task}.json", result)
            # Inputs survive until all ten metric files have passed offline audit.
            log.info(f"[worldcache-benchmark] {task}: speedup={result['speedup']:.3f}x")


def main():
    from cosmos_framework.inference.common.args import tyro_cli

    args = tyro_cli(Args, description=__doc__)
    service = Service(args)
    if args.eval_phase == "benchmark":
        service.benchmark()
        return
    log.info(f"[worldcache-server] READY {args.host}:{args.port}")
    _load_openpi_websocket_policy_server()(
        policy=service, host=args.host, port=args.port, metadata={"strategy": "worldcache_dddc_joint"}
    ).serve_forever()


if __name__ == "__main__":
    main()
