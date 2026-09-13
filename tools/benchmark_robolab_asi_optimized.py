"""Paired, warm generation benchmark using one loaded RoboLab Edge model."""

from cosmos_framework.inference.common.init import init_script

init_script()

import argparse
import copy
import hashlib
import json
import random
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.inference.edge_core_stable import Version1Controller
from cosmos_framework.model.generator.mot.parallelize_vfm_network import apply_compile as compile_heads
from cosmos_framework.scripts.action_policy_server_robolab_version1 import Version1PolicyService, Version1ServerArgs
from cosmos_framework.scripts.robolab_version1 import Version1Controller as LegacyController


def metric(x, y):
    x, y = x.double().flatten(), y.double().flatten()
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise FloatingPointError("Nonfinite output")
    return dict(
        mse=(x - y).square().mean().item(),
        max_abs=(x - y).abs().max().item(),
        relative_l2=((x - y).norm() / y.norm().clamp_min(1e-24)).item(),
        cosine=(torch.dot(x, y) / (x.norm() * y.norm()).clamp_min(1e-24)).item(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--vae", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument(
        "--modes",
        nargs="+",
        default=["dense", "legacy", "optimized-eager", "compile", "compile-graph"],
        choices=["dense", "legacy", "optimized-eager", "compile", "compile-graph"],
    )
    args = p.parse_args()
    if args.warmup < 3 or args.repeats < 1:
        p.error("Use at least 3 warmups and 1 measured repeat")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    capture = torch.load(args.capture, map_location="cpu", weights_only=False)
    if capture["kwargs"]["num_steps"] != 4 or capture["kwargs"]["shift"] != 5 or capture["kwargs"]["guidance"] != 3:
        raise ValueError("Requires 4 steps / shift5 / CFG3")
    service = Version1PolicyService(
        Version1ServerArgs(
            checkpoint_path=str(args.checkpoint),
            asi_execution="legacy",
            seed=0,
            deterministic_seed=False,
            guidance=3,
            num_steps=4,
            shift=5,
            guardrails=False,
            format_prompt_as_json=True,
            output_dir=args.output / "model_output",
            experiment_overrides=[
                f"model.config.tokenizer.vae_path={args.vae}",
                "model.config.tokenizer.object_store_credential_path_pretrained=",
                "model.config.tokenizer.bucket_name=",
            ],
        )
    )
    # Unwrap only the service's request controller, retaining the loaded native model.
    model = service.model
    generate = type(model).generate_samples_from_batch.__get__(model)
    net = model.net
    eager_layers = list(net.language_model.model.layers)
    head_names = ("_encode_text", "_encode_vision", "_encode_action", "_decode_vision", "_decode_action")
    eager_heads = {k: getattr(net, k) for k in head_names}
    implementations = {}
    for mode in args.modes:
        if mode not in ("compile", "compile-graph"):
            implementations[mode] = (eager_layers, eager_heads)
            continue
        graph = mode == "compile-graph"
        cfg = CompileConfig(
            enabled=True,
            compiled_region="all",
            use_cuda_graphs=graph,
            compile_dynamic=model.config.compile.compile_dynamic,
        )
        layers = [
            torch.compile(
                layer, fullgraph=True, dynamic=cfg.compile_dynamic, mode="reduce-overhead" if graph else "default"
            )
            for layer in eager_layers
        ]
        for k, v in eager_heads.items():
            setattr(net, k, v)
        compile_heads(net, cfg)
        implementations[mode] = (layers, {k: getattr(net, k) for k in head_names})

    def run(mode, seed_offset=0):
        layers, heads = implementations[mode]
        for i, layer in enumerate(layers):
            net.language_model.model.layers[i] = layer
        for k, v in heads.items():
            setattr(net, k, v)
        net.pad_for_cuda_graphs = mode == "compile-graph"
        positional, kwargs = copy.deepcopy(capture["args"]), copy.deepcopy(capture["kwargs"])
        kwargs["seed"] = [int(kwargs["seed"][0]) + seed_offset]
        seed = kwargs["seed"][0]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ctrl = None
        if mode != "dense":
            cls = LegacyController if mode == "legacy" else Version1Controller
            extra = (
                {}
                if mode == "legacy"
                else dict(
                    optimized=True,
                    cache_layout=True,
                    compile_profile_decoder=True,
                    compile_profile_kernel=False,
                    cuda_graphs=mode == "compile-graph",
                )
            )
            ctrl = cls(torch=torch, net=net, guidance=3, num_steps=4, **extra)
        with torch.inference_mode(), ctrl if ctrl is not None else nullcontext():
            torch.cuda.synchronize()
            start = time.perf_counter()
            samples = generate(*positional, **kwargs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        # Validation, artifact copies and summaries are outside measured generation.
        summary = ctrl.finish() if ctrl else {}
        result = {k: samples[k][0].detach().cpu().clone() for k in ("action", "vision")}
        result["predicted_action"] = result["action"][service.cfg.history_length :, : service.cfg.action_dim].clone()
        if service.cfg.action_space != "joint_pos":
            raise ValueError("This RoboLab benchmark requires joint_pos")
        result["predicted_action"][:, -1] = 1.0 - result["predicted_action"][:, -1]
        result["future_vision"] = result["vision"][:, :, 1:].clone()
        for value in result.values():
            if not torch.isfinite(value).all():
                raise FloatingPointError(mode)
        if ctrl:
            result.update(
                {k: ctrl.plan[k].detach().cpu().clone() for k in ("core_masks", "stable_mask", "execution_mask")}
            )
            result["core_blocks"] = ctrl.plan["core_blocks"]
            result["profiles"] = torch.stack([r["profiles"] for r in ctrl._profile_records]).detach().cpu().clone()
        return result, elapsed, summary

    report = dict(
        python=sys.executable,
        source=str(Path(__file__).resolve()),
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        capture=str(args.capture),
        capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
        timing_boundary="generate_samples_from_batch + CUDA synchronize; no decode/RPC/summary/copies",
        compile_dynamic=model.config.compile.compile_dynamic,
        inductor_precision_casts=torch._inductor.config.emulate_precision_casts,
        inductor_division_rounding=torch._inductor.config.emulate_divison_rounding,
        capture_metadata=capture.get("metadata"),
        modes={},
        comparisons={},
    )
    latest = {}
    for mode in args.modes:
        for i in range(args.warmup):
            result, elapsed, summary = run(mode)
            print(f"WARMUP {mode} {i + 1}/{args.warmup} {elapsed:.6f}s", flush=True)
        latest[mode] = result
        report["modes"][mode] = dict(warm_samples_s=[], summary=summary)
    for rep in range(args.repeats):
        order = args.modes if rep % 2 == 0 else list(reversed(args.modes))
        for mode in order:
            result, elapsed, summary = run(mode)
            # Fixed input must not change on subsequent graph replay.
            for key in ("action", "vision"):
                torch.testing.assert_close(result[key], latest[mode][key], rtol=1e-5, atol=1e-5)
            if mode != "dense":
                for key in ("core_masks", "stable_mask", "execution_mask"):
                    if not torch.equal(result[key], latest[mode][key]):
                        raise AssertionError(f"{mode}: mask changed on same-input replay")
            report["modes"][mode]["warm_samples_s"].append(elapsed)
            print(f"TIMING {rep + 1}/{args.repeats} {mode} {elapsed:.6f}s", flush=True)
    for mode, row in report["modes"].items():
        times = row["warm_samples_s"]
        row.update(
            mean_s=statistics.mean(times), median_s=statistics.median(times), p90_s=float(np.percentile(times, 90))
        )
        torch.save(latest[mode], args.output / f"{mode}.pt")
    reference = "legacy" if "legacy" in latest else args.modes[0]
    for mode in args.modes:
        report["comparisons"][mode] = {
            k: metric(latest[mode][k], latest[reference][k])
            for k in ("action", "vision", "predicted_action", "future_vision")
        }
        if mode != "dense" and reference != "dense":
            report["comparisons"][mode]["profiles"] = metric(latest[mode]["profiles"], latest[reference]["profiles"])
            report["comparisons"][mode]["mask_exact"] = all(
                torch.equal(latest[mode][k], latest[reference][k])
                for k in ("core_masks", "stable_mask", "execution_mask")
            )
            report["comparisons"][mode]["mask_changed_tokens"] = int(
                (latest[mode]["execution_mask"] != latest[reference]["execution_mask"]).sum()
            )
            report["comparisons"][mode]["core_blocks_exact"] = (
                latest[mode]["core_blocks"] == latest[reference]["core_blocks"]
            )
    # A changed input seed after warmup catches stale graph result reuse.
    report["changed_seed"] = {}
    changed = {mode: run(mode, seed_offset=1)[0] for mode in args.modes}
    for mode in args.modes:
        if torch.equal(changed[mode]["vision"], latest[mode]["vision"]):
            raise AssertionError(f"{mode}: changed seed produced stale latent")
        report["changed_seed"][mode] = {
            k: metric(changed[mode][k], changed[reference][k])
            for k in ("action", "vision", "predicted_action", "future_vision")
        }
        if mode != "dense" and reference != "dense":
            report["changed_seed"][mode]["mask_changed_tokens"] = int(
                (changed[mode]["execution_mask"] != changed[reference]["execution_mask"]).sum()
            )
    report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
