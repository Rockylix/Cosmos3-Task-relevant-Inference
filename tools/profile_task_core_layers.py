"""Paired native chunk timing and separate, instrumented Core/Stable selection profiling."""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import inspect
import itertools
import json
import os
import random
import subprocess
import textwrap
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODES = ("every_chunk_top6", "every_chunk_top7", "task_fixed_top7")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    assert np.isfinite(values).all()
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p90=float(np.quantile(values, .9)), min=float(values.min()), max=float(values.max()))


class StageTimer:
    """CUDA stream elapsed includes launch gaps, not just active CUDA kernel time."""

    def __init__(self):
        self.records = []

    @contextmanager
    def scope(self, name):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        begin = time.perf_counter()
        try:
            yield
        finally:
            wall_ms = (time.perf_counter() - begin) * 1000
            end.record()
            self.records.append((name, wall_ms, start, end))

    def collect(self):
        torch.cuda.synchronize()
        return [dict(stage=n, host_scope_ms=w, cuda_stream_ms=a.elapsed_time(b))
                for n, w, a, b in self.records]


def instrument_plan(original, timer):
    """Wrap the original AST statements in timing scopes without rewriting any expression."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    fn = tree.body[0]
    boundaries = {
        "eps": "rhq_all28",
        "core_quality": "core_weighted_score",
        "global_weights": "stable_global_score",
        "core_pool_budget": "token_topk_and_validation",
    }
    stage, groups = "stack_validate", []
    statements = []
    for node in fn.body:
        next_stage = None
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            next_stage = boundaries.get(node.targets[0].id)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "fixed_core_blocks is None":
            next_stage = "core_layer_ids"
        if isinstance(node, ast.Return):
            next_stage = "return_plan"
        if next_stage is not None and next_stage != stage:
            groups.append((stage, statements))
            stage, statements = next_stage, []
        statements.append(node)
    groups.append((stage, statements))
    assert [n for n, _ in groups] == ["stack_validate", "rhq_all28", "core_layer_ids",
        "core_weighted_score", "stable_global_score", "token_topk_and_validation", "return_plan"]
    fn.body = [ast.With(items=[ast.withitem(context_expr=ast.Call(
        func=ast.Name(id="_stage_scope", ctx=ast.Load()), args=[ast.Constant(name)], keywords=[]))],
        body=body) for name, body in groups]
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__, _stage_scope=timer.scope)
    exec(compile(tree, "<timed-original-build-core-stable-plan>", "exec"), namespace)
    return namespace[original.__name__]


@contextmanager
def profile_scopes(module, timer):
    # Diagnostic runs ONLY. The primary benchmark never replaces either callable.
    plan, profile = module.build_core_stable_plan, module.action_aligned_future_profiles_with_lse
    timed_plan = instrument_plan(plan, timer)

    def plan_wrapper(**kwargs):
        with timer.scope("selection_plan_total"):
            return timed_plan(**kwargs)

    def profile_wrapper(**kwargs):
        with timer.scope("action_attention_profile"):
            return profile(**kwargs)

    module.build_core_stable_plan = plan_wrapper
    module.action_aligned_future_profiles_with_lse = profile_wrapper
    try:
        yield
    finally:
        module.build_core_stable_plan = plan
        module.action_aligned_future_profiles_with_lse = profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--profile-rounds", type=int, default=6)
    parser.add_argument("--primary-from", type=Path, help="Reuse a complete native timing CSV after a diagnostic-only failure")
    args = parser.parse_args()
    assert args.rounds >= 6 and args.rounds % 6 == 0
    assert args.profile_rounds >= 6 and args.profile_rounds % 6 == 0
    assert torch.cuda.is_available()
    from cosmos_framework.inference.task_core_layers import TaskCoreLayerCache
    from cosmos_framework.scripts.asi_velocity_cache_chunk import CAPTURE, VAE
    from cosmos_framework.scripts.asi_velocity_cache_smoke import Args, Service
    from cosmos_framework.scripts import robolab_version1 as version1

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    task_run = ROOT / "experiments/task_core_top7_global_stable_smoke3_s0_p0_v1"
    task_rows = [json.loads(line) for line in (task_run / "requests.jsonl").read_text().splitlines()]
    first = next(r for r in task_rows if r["task"] == "BananaInBowlTask" and r["chunk"] == 1)
    fixed = tuple(first["core_blocks"])
    source_paths = ["cosmos_framework/scripts/robolab_version1.py",
        "cosmos_framework/scripts/asi_velocity_cache_smoke.py", "cosmos_framework/inference/task_core_layers.py",
        "cosmos_framework/inference/asi_velocity_cache.py", "tools/profile_task_core_layers.py"]
    write_json(output / "manifest.json", dict(
        tasks=["BananaInBowlTask"], task_core_top7=True, task="BananaInBowlTask", input=str(CAPTURE),
        input_sha256=hashlib.sha256(CAPTURE.read_bytes()).hexdigest(), captured_chunk=3,
        fixed_layer_ids=fixed, fixed_ids_source=str(task_run / "requests.jsonl"),
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths},
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), pythonpath=os.environ.get("PYTHONPATH"),
        compile=False, cuda_graphs=False, shift=5, steps=4, guidance=3,
        rounds=args.rounds, warmups_per_mode=args.warmups, profile_rounds=args.profile_rounds,
        timing="Service.generate_fast synchronized generation wall; excludes model load, input cloning, VAE, RPC, CSV",
        profile="separate diagnostic runs; CUDA event elapsed includes CPU launch gaps; not active kernel duration",
        schedule="all modes one dense conditional + seven sparse; all28 profile; Core80 Stable104; ASI velocity cache",
        primary_from=None if args.primary_from is None else str(args.primary_from.resolve()),
    ))
    service = Service(Args(eval_root=output, task_core_top7=True,
        checkpoint_path="/root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID", guardrails=False,
        output_dir=output / "model_output", experiment_overrides=[f"model.config.tokenizer.vae_path={VAE}",
        "model.config.tokenizer.object_store_credential_path_pretrained=", "model.config.tokenizer.bucket_name="]))
    capture = torch.load(CAPTURE, weights_only=False, map_location="cpu")
    seed = capture["kwargs"]["seed"][0]
    references = {}

    def run(mode):
        positional, kwargs = copy.deepcopy(capture["args"]), copy.deepcopy(capture["kwargs"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cache = None
        if mode != MODES[0]:
            cache = TaskCoreLayerCache(count=7)
            cache.begin_task("BananaInBowlTask")
            if mode == MODES[2]:
                cache.commit(fixed)
        samples, elapsed, summary, controller, sampler = service.generate_fast(positional, kwargs, layer_cache=cache)
        assert summary["profiled_block_count"] == 28
        assert summary["stable_profile_blocks"] == list(range(28))
        assert (summary["dense_stack_count"], summary["sparse_stack_count"]) == (1, 7)
        assert sampler.evaluations == 4
        if mode == MODES[2]:
            assert tuple(summary["core_blocks"]) == fixed and summary["core_layers_reused"]
        return samples, elapsed * 1000, summary, controller

    # References/cold start and warmup excluded from every reported primary sample.
    for mode in MODES:
        samples, _, _, controller = run(mode)
        references[mode] = {k: samples[k][0].detach().cpu().clone() for k in ("action", "vision")}
        references[mode]["mask"] = controller.plan["execution_mask"].cpu().clone()
    for _ in range(args.warmups):
        for mode in MODES:
            run(mode)
    orders = list(itertools.permutations(MODES))
    primary = []
    if args.primary_from is None:
        for round_id in range(args.rounds):
            for order_id, mode in enumerate(orders[round_id % len(orders)]):
                samples, ms, summary, controller = run(mode)
                primary.append(dict(round=round_id, order=order_id, mode=mode, generation_ms=ms,
                    profiled_blocks=summary["profiled_block_count"], core_layers_reused=summary["core_layers_reused"]))
            if (round_id + 1) % 6 == 0:
                print(f"[timing] native rounds={round_id+1}/{args.rounds}", flush=True)
    else:
        old_manifest = json.loads((args.primary_from / "manifest.json").read_text())
        manifest = json.loads((output / "manifest.json").read_text())
        for key in ("input_sha256", "fixed_layer_ids", "torch", "gpu", "rounds", "shift", "steps", "guidance"):
            assert old_manifest[key] == manifest[key], key
        for name, sha in old_manifest["source_sha256"].items():
            if name != "tools/profile_task_core_layers.py":
                assert manifest["source_sha256"][name] == sha, name
        source = args.primary_from / "chunk_timing.csv"
        with source.open() as stream:
            for row in csv.DictReader(stream):
                primary.append(dict(round=int(row["round"]), order=int(row["order"]), mode=row["mode"],
                    generation_ms=float(row["generation_ms"]), profiled_blocks=int(row["profiled_blocks"]),
                    core_layers_reused=row["core_layers_reused"] == "True"))
        assert len(primary) == args.rounds * 3
        assert len({(r["round"],r["mode"]) for r in primary}) == len(primary)
        manifest["primary_csv_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest["primary_recovery_note"] = "All native samples completed; diagnostic AST segmentation assertion repaired; no samples discarded"
        write_json(output / "manifest.json", manifest)
        print(f"[timing] reused ALL {len(primary)} native samples from {source}", flush=True)
    write_csv(output / "chunk_timing.csv", primary)
    diagnostics = []
    for round_id in range(args.profile_rounds):
        for mode in orders[round_id % len(orders)]:
            timer = StageTimer()
            with profile_scopes(version1, timer):
                samples, ms, summary, controller = run(mode)
            measured = timer.collect()
            assert sum(r["stage"] == "action_attention_profile" for r in measured) == 28
            for key in ("action", "vision"):
                torch.testing.assert_close(samples[key][0].cpu(), references[mode][key], rtol=0, atol=0)
            torch.testing.assert_close(controller.plan["execution_mask"].cpu(), references[mode]["mask"], rtol=0, atol=0)
            for record in measured:
                diagnostics.append(dict(round=round_id, mode=mode, **record))
        print(f"[profile] diagnostic rounds={round_id+1}/{args.profile_rounds} exact-output checks PASS", flush=True)
    write_csv(output / "stage_scopes.csv", diagnostics)
    # Additional same-profile replay isolates selection timing from transformer variability.
    records = [{"block": r["block"], "profiles": r["profiles"].clone()} for r in controller._profile_records]
    micro = []
    with torch.inference_mode():
        for round_id in range(210):
            for mode in orders[round_id % len(orders)]:
                options = dict(torch=torch, profile_records=records,
                    core_block_count=6 if mode == MODES[0] else 7,
                    fixed_core_blocks=fixed if mode == MODES[2] else None)
                torch.cuda.synchronize()
                begin = time.perf_counter()
                version1.build_core_stable_plan(**options)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - begin) * 1000
                if round_id >= 10:
                    micro.append(dict(round=round_id-10, mode=mode, selection_ms=ms))
    write_csv(output / "selection_microbench.csv", micro)
    timing = {m: stats([r["generation_ms"] for r in primary if r["mode"] == m]) for m in MODES}
    selection = {m: stats([r["selection_ms"] for r in micro if r["mode"] == m]) for m in MODES}
    stage_totals = {}
    for mode in MODES:
        stage_totals[mode] = {}
        for stage in sorted(set(r["stage"] for r in diagnostics)):
            per_round = [dict(host=sum(r["host_scope_ms"] for r in diagnostics
                if r["round"] == i and r["mode"] == mode and r["stage"] == stage),
                cuda=sum(r["cuda_stream_ms"] for r in diagnostics
                if r["round"] == i and r["mode"] == mode and r["stage"] == stage))
                for i in range(args.profile_rounds)]
            stage_totals[mode][stage] = {"host_ms":stats([r["host"] for r in per_round]),
                "cuda_stream_ms":stats([r["cuda"] for r in per_round])}
    paired = {}
    for other in MODES[:2]:
        times = {m: np.array([r["generation_ms"] for r in primary if r["mode"] == m]) for m in MODES}
        delta = times[other] - times[MODES[2]]
        rng = np.random.default_rng(20260911)
        boot = np.median(delta[rng.integers(0, len(delta), (10000, len(delta)))], axis=1)
        paired[other+"_minus_fixed"] = {"delta_ms": stats(delta),
            "paired_median_bootstrap_95ci_ms": np.quantile(boot,[.025,.975]).tolist(),
            "median_speed_ratio":timing[other]["median"]/timing[MODES[2]]["median"]}
    result = dict(passed=True, timing_ms=timing, selection_microbench_ms=selection, stages=stage_totals,
        paired_comparison=paired, profile_matches_native_exact=True, all_modes_profile_count=28,
        no_nsys_capture=True, primary_has_no_stage_instrumentation=True,
        qualification="One stored chunk on one GPU; no closed-loop success or image quality comparison")
    write_json(output / "metrics.json", result)
    print(json.dumps(dict(timing_ms=timing, paired_comparison=paired), indent=2), flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
