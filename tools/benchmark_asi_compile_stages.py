"""Cumulative compiler evaluation on B6 main-LSE ASI; never changes server defaults."""

from cosmos_framework.inference.common.init import init_script

init_script()

import argparse
import copy
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.inference.asi_main_lse import MainLSEController
from cosmos_framework.model.generator.mot.parallelize_vfm_network import apply_compile as compile_heads
from cosmos_framework.scripts.action_policy_server_robolab_version1 import Version1PolicyService, Version1ServerArgs
from tools.benchmark_asi_batch_only import CAPTURE, ROOT, VAE, compare

MODES = ("eager", "sparse_compile", "decoder_compile", "all_compile", "all_graph")
HEADS = ("_encode_text", "_encode_vision", "_encode_action", "_decode_vision", "_decode_action")


class StagedController(MainLSEController):
    """Keep first dense stack eager without reverting metadata/LSE scoring."""

    dense_eager = False
    trace = False

    def scope(self, label):
        return torch.cuda.nvtx.range(label) if self.trace else nullcontext()

    def _run_dense_profile_layer(self, **kwargs):
        if self.dense_eager:
            layer = kwargs["decoder_layer"]
            kwargs["decoder_layer"] = getattr(layer, "_orig_mod", layer)
        return super()._run_dense_profile_layer(**kwargs)

    def run_layer(self, **kwargs):
        with self.scope(f"asi.step{self._current['step']}.{self._current['branch']}.B{kwargs['block']:02d}"):
            return super().run_layer(**kwargs)

    def _slice_pack_and_rope(self, *args, **kwargs):
        with self.scope("asi.pack"):
            return super()._slice_pack_and_rope(*args, **kwargs)

    def end_stack(self, *args, **kwargs):
        with self.scope("asi.select" if self._current == dict(step=0, branch="conditional") else "asi.restore"):
            return super().end_stack(*args, **kwargs)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str))


def check_repeat(current, previous):
    diff = compare(current, previous)
    if not (diff["action"]["exact"] and diff["future_vision"]["exact"] and diff["execution_mask_xor"] == 0):
        raise RuntimeError(f"Same-mode repeat changed: {diff}")
    return diff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", type=Path, default=CAPTURE)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--trace-mode", choices=MODES)
    args = parser.parse_args()
    if args.warmup < 5 or args.repeats < 1 or (not args.trace_mode and "eager" not in args.modes):
        parser.error("Require >=5 warmups, >=1 repeats, and eager reference")
    args.output.mkdir(parents=True, exist_ok=False)
    capture = torch.load(args.capture, map_location="cpu", weights_only=False)
    assert all(capture["kwargs"][k] == v for k, v in dict(num_steps=4, guidance=3, shift=5).items())
    service = Version1PolicyService(
        Version1ServerArgs(
            checkpoint_path="/root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID",
            asi_execution="legacy",
            num_steps=4,
            guidance=3,
            shift=5,
            seed=0,
            deterministic_seed=False,
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
    model, net = service.model, service.model.net
    generate = type(model).generate_samples_from_batch.__get__(model)
    eager_layers = list(net.language_model.model.layers)
    eager_heads = {k: getattr(net, k) for k in HEADS}
    assert not any(hasattr(layer, "_orig_mod") for layer in eager_layers)
    # Same underlying frozen weights; wrappers specialize execution, not parameters.
    layer_sets, head_sets = {}, {}

    def install(mode):
        compiled = mode != "eager"
        graph = mode == "all_graph"
        key = "graph" if graph else "compile"
        if compiled and key not in layer_sets:
            layer_sets[key] = [
                torch.compile(
                    layer,
                    fullgraph=True,
                    dynamic=model.config.compile.compile_dynamic,
                    mode="reduce-overhead" if graph else "default",
                )
                for layer in eager_layers
            ]
        layers = layer_sets[key] if compiled else eager_layers
        for index, layer in enumerate(layers):
            net.language_model.model.layers[index] = layer
        heads_compiled = mode in ("all_compile", "all_graph")
        if heads_compiled and key not in head_sets:
            for name, fn in eager_heads.items():
                setattr(net, name, fn)
            compile_heads(
                net,
                CompileConfig(
                    enabled=True,
                    compiled_region="all",
                    use_cuda_graphs=graph,
                    compile_dynamic=model.config.compile.compile_dynamic,
                ),
            )
            head_sets[key] = {name: getattr(net, name) for name in HEADS}
        for name, fn in (head_sets[key] if heads_compiled else eager_heads).items():
            setattr(net, name, fn)
        # Keep actual lengths fixed: adding Graph must not also introduce padding.
        net.pad_for_cuda_graphs = False

    def run(mode, offset=0, trace=False):
        install(mode)
        positional, kwargs = copy.deepcopy(capture["args"]), copy.deepcopy(capture["kwargs"])
        seed = int(kwargs["seed"][0]) + offset
        kwargs["seed"] = [seed]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ctrl = StagedController(torch=torch, net=net, guidance=3, num_steps=4)
        ctrl.dense_eager, ctrl.trace = mode == "sparse_compile", trace
        with torch.inference_mode(), ctrl:
            torch.cuda.synchronize()
            if trace:
                torch.cuda.cudart().cudaProfilerStart()
                torch.cuda.nvtx.range_push("asi.chunk.generate")
            start = time.perf_counter()
            try:
                samples = generate(*positional, **kwargs)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
            finally:
                if trace:
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.cudart().cudaProfilerStop()
        summary = ctrl.finish()
        result = {
            k: ctrl.plan[k].detach().cpu().clone()
            for k in (
                "raw_profiles",
                "block_mass",
                "block_entropy",
                "block_quality",
                "core_masks",
                "stable_mask",
                "execution_mask",
            )
        }
        result["core_blocks"] = ctrl.plan["core_blocks"]
        result["action"] = (
            samples["action"][0][service.cfg.history_length :, : service.cfg.action_dim].detach().cpu().clone()
        )
        result["action"][:, -1] = 1 - result["action"][:, -1]
        result["future_vision"] = samples["vision"][0][:, :, 1:].detach().cpu().clone()
        if any(not torch.isfinite(value).all() for value in result.values() if torch.is_tensor(value)):
            raise FloatingPointError(mode)
        assert result["execution_mask"].sum(-1).tolist() == [184] * 8
        assert (summary["dense_stack_count"], summary["sparse_stack_count"]) == (1, 7)
        summary.update(
            dense_profile_eager=ctrl.dense_eager,
            main_lse_reuse=True,
            heads_compiled=mode in ("all_compile", "all_graph"),
            graph_requested=mode == "all_graph",
            expected_compiled_decoder_calls=0 if mode == "eager" else (196 if mode == "sparse_compile" else 224),
        )
        return result, elapsed, summary

    files = [
        "tools/benchmark_asi_compile_stages.py",
        "cosmos_framework/inference/asi_main_lse.py",
        "cosmos_framework/inference/asi_existing_ablation.py",
        "cosmos_framework/inference/edge_core_stable.py",
        "cosmos_framework/inference/edge_core_stable_fast.py",
        "cosmos_framework/model/generator/mot/unified_mot.py",
        "cosmos_framework/model/generator/mot/attention.py",
        "cosmos_framework/model/generator/mot/inference_text_kv_memory.py",
    ]
    report = dict(
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
        capture=str(args.capture),
        capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
        python=sys.executable,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        seed=capture["kwargs"]["seed"],
        parameters=dict(shift=5, steps=4, guidance=3, core=80, stable=104, main_lse=True),
        compile_dynamic=model.config.compile.compile_dynamic,
        pad_for_cuda_graphs=False,
        warmup={},
        comparisons={},
        summaries={},
        timing={},
        timing_scope="synchronized generate_samples_from_batch, fresh input/RNG/controller; no CPU summaries/decode/RPC; profiler separate",
        acceptance="quantify compiled numerical differences; no production-equivalence assertion",
        trace_mode=args.trace_mode,
    )
    save(args.output / "manifest.json", report)
    modes = [args.trace_mode] if args.trace_mode else args.modes
    latest = {}
    try:
        for mode in modes:
            report["warmup"][mode] = []
            for i in range(args.warmup):
                latest[mode] = run(mode)
                report["warmup"][mode].append(latest[mode][1])
                save(args.output / "progress.json", report)
                print("WARMUP", mode, i + 1, latest[mode][1], flush=True)
        if args.trace_mode:
            value, elapsed, summary = run(args.trace_mode, trace=True)
            report.update(
                trace_wall_s=elapsed, trace_repeat=check_repeat(value, latest[args.trace_mode][0]), summary=summary
            )
        else:
            seed0_reference = None
            for offset in (0, 1):
                paired = {m: run(m, offset)[0] for m in modes}
                if offset == 0:
                    seed0_reference = paired
                report["comparisons"][str(offset)] = {m: compare(paired[m], paired["eager"]) for m in modes}
                torch.save(paired, args.output / f"outputs_seed_offset{offset}.pt")
                save(args.output / "gate.json", report)
                if "sparse_compile" in modes:
                    diff = report["comparisons"][str(offset)]["sparse_compile"]
                    assert diff["raw_profiles"]["exact"] and diff["execution_mask_xor"] == 0, (
                        "Eager profile changed in sparse-only compile"
                    )
                for mode in modes:
                    repeat, _, _ = run(mode, offset)
                    check_repeat(repeat, paired[mode])
                    if offset == 1:
                        assert not torch.equal(repeat["future_vision"], latest[mode][0]["future_vision"]), (
                            "Stale graph/seed output"
                        )
            # Refresh every mode after seed switching, before formal alternating timing.
            report["return_to_seed0"] = {}
            for mode in modes:
                latest[mode] = run(mode)
                report["return_to_seed0"][mode] = check_repeat(latest[mode][0], seed0_reference[mode])
            save(args.output / "gate.json", report)
            times = {m: [] for m in modes}
            for rep in range(args.repeats):
                for mode in modes if rep % 2 == 0 else list(reversed(modes)):
                    value, elapsed, summary = run(mode)
                    check_repeat(value, latest[mode][0])
                    times[mode].append(elapsed)
                    report["summaries"][mode] = summary
                    print("TIMING", rep + 1, mode, elapsed, flush=True)
            for mode, values in times.items():
                report["timing"][mode] = dict(
                    samples_s=values,
                    mean_s=statistics.mean(values),
                    median_s=statistics.median(values),
                    p90_s=float(np.percentile(values, 90)),
                )
            report["speedup_vs_current_eager"] = {
                m: report["timing"]["eager"]["median_s"] / report["timing"][m]["median_s"] for m in modes
            }
        save(args.output / "results.json", report)
        print("COMPLETE", args.output, flush=True)
    except Exception as exc:
        report["error"] = repr(exc)
        save(args.output / "failure.json", report)
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
