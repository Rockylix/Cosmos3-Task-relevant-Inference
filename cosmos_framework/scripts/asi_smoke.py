"""Current ASI ten-task server and paired eager benchmark; policy code is unmodified."""

from cosmos_framework.inference.common.init import init_script

init_script()

import copy
import json
import random
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq
from cosmos_framework.scripts.action_policy_server_robolab import (
    RobolabPolicyService,
    RobolabServerArgs,
    _load_openpi_websocket_policy_server,
)
from cosmos_framework.scripts.robolab_version1 import STRATEGY_VERSION, Version1Controller
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
    x, y = test.double().reshape(-1), reference.double().reshape(-1)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise FloatingPointError("Nonfinite metric input")
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
    format_prompt_as_json: bool | None = True
    eval_root: Path
    eval_phase: Literal["serve", "benchmark"] = "serve"
    gate_capture: Path


class Service(RobolabPolicyService):
    def _build_setup_args(self, args):
        return super()._build_setup_args(args).model_copy(update={"use_torch_compile": False, "use_cuda_graphs": False})

    def __init__(self, args):
        if (
            args.num_steps,
            args.shift,
            args.guidance,
            args.seed,
            args.deterministic_seed,
            args.format_prompt_as_json,
        ) != (4, 5, 3, 0, False, True):
            raise ValueError(
                "Frozen ten-task protocol: 4 steps, shift5, CFG3, policyseed0, changing request seeds, structured prompt"
            )
        self.run = args.eval_root.resolve()
        self.manifest = json.loads((self.run / "manifest.json").read_text())
        metadata = json.loads(Path("/root/robolab/RoboLab/robolab/tasks/_metadata/task_metadata.json").read_text())
        self.task_for_prompt = {
            r["instruction"]: r["task_name"] for r in metadata if r["task_name"] in self.manifest["tasks"]
        }
        self.prompt, self.chunk = None, 0
        super().__init__(args)
        self.original_generate = self.model.generate_samples_from_batch
        import cosmos_framework.model.generator.mot.unified_mot as mot
        import cosmos_framework.scripts.robolab_version1 as policy

        write_json(
            self.run / f"runtime_{args.eval_phase}.json",
            {
                "strategy": STRATEGY_VERSION,
                "model_source": mot.__file__,
                "controller_source": policy.__file__,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "compile": False,
                "cuda_graphs": False,
                "policy_seed": 0,
                "reset_rng_each_task": True,
                "deterministic_seed": False,
            },
        )
        if args.eval_phase == "serve":
            if (self.run / "requests.jsonl").exists():
                raise FileExistsError("Refusing to overwrite existing requests")
            self.gate(torch.load(args.gate_capture, weights_only=False, map_location="cpu"))
            self.model.generate_samples_from_batch = self.generate_logged

    @torch.inference_mode()
    def generate_case(self, capture, sparse, audit=False):
        positional, kwargs = copy.deepcopy(capture["args"]), copy.deepcopy(capture["kwargs"])
        seed = kwargs["seed"][0]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        rows, finite, handles = {}, [], []
        if audit:
            for block, layer in enumerate(self.model.net.language_model.model.layers):
                modules = {n: getattr(layer.self_attn, f"{n}_proj_moe_gen") for n in ("q", "k", "v", "o")}
                modules["mlp"] = layer.mlp_moe_gen
                for name, module in modules.items():
                    key = (block, name)
                    rows[key] = []
                    handles.append(module.register_forward_pre_hook(lambda m, a, key=key: rows[key].append(len(a[0]))))
                handles.append(
                    layer.register_forward_hook(
                        lambda m, a, out: finite.append(torch.isfinite(get_gen_seq(out[0])).all())
                    )
                )
        torch.cuda.synchronize()
        started = time.perf_counter()
        controller = Version1Controller(torch=torch, net=self.model.net, guidance=3, num_steps=4) if sparse else None
        try:
            with controller if controller is not None else nullcontext():
                samples = self.original_generate(*positional, **kwargs)
            summary = controller.finish() if controller is not None else {}
        finally:
            for h in handles:
                h.remove()
        for key in ("action", "vision"):
            if not torch.isfinite(samples[key][0]).all().item():
                raise FloatingPointError(key)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if audit:
            expected = [3093] + [1845] * 7 if sparse else [3093] * 8
            for key, value in rows.items():
                if value != expected:
                    raise AssertionError((key, value, expected))
            if len(finite) != 224 or not torch.stack(finite).all().item():
                raise FloatingPointError("Bad intermediate block output or count")
        return {k: cpu_tree(samples[k][0]) for k in ("action", "vision")}, elapsed, summary

    def gate(self, capture):
        dense, _, _ = self.generate_case(capture, False, audit=True)
        sparse, _, summary = self.generate_case(capture, True, audit=True)
        for key in ("action", "vision"):
            torch.testing.assert_close(dense[key], capture["outputs"][key], rtol=1e-4, atol=1e-4)
        write_json(
            self.run / "gpu_gate.json",
            {
                "passed": True,
                "all_intermediate_finite": True,
                "actual_module_row_checks": "all 28 blocks x Q/K/V/O/MLP; Dense [3093]*8, ASI [3093]+[1845]*7",
                "dense_vs_stored_dense": {k: metrics(dense[k], capture["outputs"][k]) for k in dense},
                "asi_vs_dense": {k: metrics(sparse[k], dense[k]) for k in dense},
                "controller": summary,
                "note": "Audit timings are excluded from performance results",
            },
        )
        torch.cuda.empty_cache()
        log.info("[asi] GPU gate PASS: one dense + seven genuinely sparse stacks, all Q/K/V/O/MLP row counts checked")

    def infer(self, obs):
        prompt = obs["prompt"]
        if prompt not in self.task_for_prompt:
            raise ValueError(f"Unexpected task {prompt}")
        if prompt != self.prompt:
            self.prompt, self.chunk = prompt, 0
            self._rng = np.random.default_rng(self.cfg.seed)
        self.chunk += 1
        return super().infer(obs)

    def generate_logged(self, *positional, **kwargs):
        captured = cpu_tree({"args": positional, "kwargs": kwargs}) if self.chunk == 3 else None
        torch.cuda.synchronize()
        started = time.perf_counter()
        controller = Version1Controller(torch=torch, net=self.model.net, guidance=3, num_steps=4)
        with controller:
            samples = self.original_generate(*positional, **kwargs)
        summary = controller.finish()
        for key in ("action", "vision"):
            if not torch.isfinite(samples[key][0]).all().item():
                raise FloatingPointError(key)
        torch.cuda.synchronize()
        row = {
            "task": self.task_for_prompt[self.prompt],
            "chunk": self.chunk,
            "seed": kwargs["seed"],
            "generation_s": time.perf_counter() - started,
            "all_final_finite": True,
            **summary,
        }
        with (self.run / "requests.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if captured is not None:
            folder = self.run / "_temporary_inputs"
            folder.mkdir(exist_ok=True)
            path = folder / f"{row['task']}.pt"
            tmp = path.with_suffix(".pt.partial")
            torch.save(
                {**captured, "metadata": row, "outputs": {k: cpu_tree(samples[k][0]) for k in ("action", "vision")}},
                tmp,
            )
            tmp.replace(path)
        return samples

    @torch.inference_mode()
    def benchmark(self):
        for task in self.manifest["tasks"]:
            capture = torch.load(self.run / "_temporary_inputs" / f"{task}.pt", weights_only=False, map_location="cpu")
            replay, _, summary = self.generate_case(capture, True)
            for key in replay:
                torch.testing.assert_close(replay[key], capture["outputs"][key], rtol=1e-4, atol=1e-4)
            for i in range(5):
                for sparse in (False, True) if i % 2 == 0 else (True, False):
                    self.generate_case(capture, sparse)
            times, outputs = {"dense": [], "asi": []}, {}
            for i in range(30):
                for sparse in (False, True) if i % 2 == 0 else (True, False):
                    mode = "asi" if sparse else "dense"
                    outputs[mode], elapsed, _ = self.generate_case(capture, sparse)
                    times[mode].append(elapsed)
            rgb = {}
            for mode, out in outputs.items():
                video = self.model.decode(out["vision"].cuda()).float().cpu()
                if video.ndim != 5 or video.shape[1:3] != (3, 33):
                    raise ValueError(video.shape)
                rgb[mode] = (video[:, :, 1:].clamp(-1, 1) + 1) / 2
            timing = {
                mode: {
                    "mean_s": statistics.mean(t),
                    "median_s": statistics.median(t),
                    "p90_s": float(np.quantile(t, 0.9)),
                    "samples_s": t,
                }
                for mode, t in times.items()
            }
            write_json(
                self.run / "paired" / f"{task}.json",
                {
                    "task": task,
                    "capture": capture["metadata"],
                    "controller": summary,
                    "input_source": "ASI closed-loop chunk3, same input and noise for both methods",
                    "warmups_each": 5,
                    "repeats_each": 30,
                    "fidelity_precision": "fp64",
                    "closed_loop_replay_check_passed": True,
                    "timing": timing,
                    "speedup": timing["dense"]["median_s"] / timing["asi"]["median_s"],
                    "action": metrics(outputs["asi"]["action"][1:], outputs["dense"]["action"][1:]),
                    "future_latent": metrics(outputs["asi"]["vision"][:, :, 1:], outputs["dense"]["vision"][:, :, 1:]),
                    "future_rgb": metrics(rgb["asi"], rgb["dense"]),
                },
            )
            # All inputs survive until the complete result set has passed audit.
            log.info(f"[asi-benchmark] {task} completed")


def main():
    from cosmos_framework.inference.common.args import tyro_cli

    args = tyro_cli(Args, description=__doc__)
    service = Service(args)
    if args.eval_phase == "benchmark":
        service.benchmark()
        return
    log.info(f"[asi-server] READY {args.host}:{args.port}")
    _load_openpi_websocket_policy_server()(
        policy=service, host=args.host, port=args.port, metadata={"strategy": STRATEGY_VERSION}
    ).serve_forever()


if __name__ == "__main__":
    main()
