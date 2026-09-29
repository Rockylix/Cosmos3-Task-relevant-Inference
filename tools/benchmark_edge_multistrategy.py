"""Matched Edge benchmark: pristine Dense eager and B6 ASI/native cache strategies."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

ROOT = Path("/root/robolab")
ASI = ROOT / "cosmos-framework-edge-core80-stable104-action-weighted"
TREES = {
    "dense": ROOT / "cosmos-framework-edge",
    "asi": Path(__file__).resolve().parents[1],
    **{
        k: ROOT / "worktrees" / v
        for k, v in [("toca", "toca-future"), ("worldcache", "worldcache"), ("c3ache", "c3ache")]
    },
}
CAPTURE = ASI / "experiments/asi_smoke10_s0_p0_v1/_temporary_inputs/BananaInBowlTask.pt"
VAE = "/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def worker(args):
    from cosmos_framework.inference.common.init import init_script

    init_script()
    import random

    import numpy as np
    import torch

    import cosmos_framework
    from cosmos_framework.configs.base.defaults.compile import CompileConfig
    from cosmos_framework.model.generator.mot.parallelize_vfm_network import apply_compile
    from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService, RobolabServerArgs

    assert Path(cosmos_framework.__path__[0]).resolve() == TREES[args.worker] / "cosmos_framework"
    torch.set_num_threads(4)
    torch._dynamo.config.recompile_limit = 256
    torch._dynamo.config.accumulated_recompile_limit = 4096
    out = args.output / args.worker
    out.mkdir(parents=True, exist_ok=False)

    class Service(RobolabPolicyService):
        def _build_setup_args(self, settings):
            return (
                super()
                ._build_setup_args(settings)
                .model_copy(update={"use_torch_compile": False, "use_cuda_graphs": False})
            )

    service = Service(
        RobolabServerArgs(
            checkpoint_path=str(ROOT / "RoboLab/Cosmos3-Edge-Policy-DROID"),
            seed=0,
            deterministic_seed=False,
            guidance=3,
            num_steps=4,
            shift=5,
            format_prompt_as_json=True,
            guardrails=False,
            output_dir=out / "model_output",
            experiment_overrides=[
                f"model.config.tokenizer.vae_path={VAE}",
                "model.config.tokenizer.object_store_credential_path_pretrained=",
                "model.config.tokenizer.bucket_name=",
            ],
        )
    )
    model, net = service.model, service.model.net
    capture_path = args.capture.resolve()
    capture = torch.load(capture_path, map_location="cpu", weights_only=False)
    assert {k: capture["kwargs"][k] for k in ("guidance", "num_steps", "shift")} == {
        "guidance": 3.0,
        "num_steps": 4,
        "shift": 5.0,
    }
    input_hash = hashlib.sha256(capture_path.read_bytes()).hexdigest()
    config, cache = None, None
    if args.worker == "asi":
        from cosmos_framework.inference.asi_main_lse import MainLSEController
    elif args.worker == "toca":
        from cosmos_framework.inference.toca_future import ToCaFutureConfig, ToCaFutureController

        config = json.loads((TREES["toca"] / "configs/toca_baseline.json").read_text())["config"]
        native_config = ToCaFutureConfig(**dict(config, full_steps=tuple(config["full_steps"])))
    elif args.worker == "worldcache":
        from cosmos_framework.inference.worldcache import WorldCacheConfig

        config = json.loads((TREES["worldcache"] / "configs/worldcache_baseline.json").read_text())["config"]
        native_config = WorldCacheConfig(**config)
    elif args.worker == "c3ache":
        from cosmos_framework.inference.c3ache import C3acheCache, C3acheConfig

        config = json.loads((TREES["c3ache"] / "configs/c3ache_baseline.json").read_text())["config"]
        cache = C3acheCache(C3acheConfig(**config))

    heads = ("_encode_text", "_encode_vision", "_encode_action", "_decode_vision", "_decode_action")
    eager_layers = list(net.language_model.model.layers)
    eager_heads = {k: getattr(net, k) for k in heads}
    cfg = CompileConfig(
        enabled=True, compiled_region="all", use_cuda_graphs=True, compile_dynamic=model.config.compile.compile_dynamic
    )
    if args.worker != "dense":
        graph_layers = [
            torch.compile(layer, fullgraph=True, dynamic=cfg.compile_dynamic, mode="reduce-overhead")
            for layer in eager_layers
        ]
        apply_compile(net, cfg)
        graph_heads = {k: getattr(net, k) for k in heads}
    else:
        graph_layers, graph_heads = eager_layers, eager_heads
    selection_audit = {}

    def metric(a, b):
        a, b = a.double().flatten(), b.double().flatten()
        return dict(
            mse=(a - b).square().mean().item(),
            relative_l2=((a - b).norm() / b.norm().clamp_min(1e-24)).item(),
            cosine=((a @ b) / (a.norm() * b.norm()).clamp_min(1e-24)).item(),
        )

    def run(mode, chunk=0, seed_offset=0):
        graph = mode.endswith("graph")
        layers, hds = (graph_layers, graph_heads) if graph else (eager_layers, eager_heads)
        for i, layer in enumerate(layers):
            net.language_model.model.layers[i] = layer
        for k, v in hds.items():
            setattr(net, k, v)
        net.pad_for_cuda_graphs = False  # all strategies use actual token lengths
        # Keep public configuration truthful as well as installing compiled modules;
        # in particular do not bypass branch-native eager-only safety checks.
        model.config.compile.enabled = graph
        model.config.compile.use_cuda_graphs = graph
        positional, kwargs = copy.deepcopy(capture["args"]), copy.deepcopy(capture["kwargs"])
        kwargs["seed"] = [kwargs["seed"][0] + seed_offset]
        random.seed(kwargs["seed"][0])
        np.random.seed(kwargs["seed"][0])
        torch.manual_seed(kwargs["seed"][0])
        torch.cuda.manual_seed_all(kwargs["seed"][0])
        ctrl, summary = None, {}
        strategy = not mode.startswith("dense")
        if strategy and args.worker == "asi":
            ctrl = MainLSEController(torch=torch, net=net, num_steps=4, guidance=3)
        elif strategy and args.worker == "toca":
            extra = (
                dict(optimized=True, use_compile=graph, cuda_graphs=graph)
                if args.adapted and mode != "strategy_eager"
                else {}
            )
            ctrl = ToCaFutureController(net, native_config, **extra)
        elif strategy and args.worker == "worldcache":
            kwargs["worldcache_config"] = (
                replace(native_config, compile_compatible=True)
                if args.adapted and mode != "strategy_eager"
                else native_config
            )
        # Construction and CPU input copies outside generation; context, cache lookup,
        # profiling, selection and all GPU work required per chunk remain inside.
        torch.cuda.synchronize()
        start = time.perf_counter()
        if strategy and cache is not None:
            with cache.request(
                dict(session_id="paired", episode_id="fixed-input", chunk_id=chunk),
                signature=dict(capture_sha256=input_hash, steps=4, guidance=3, shift=5),
                transformer=net.language_model.model,
                net=net,
                branches=("conditional", "unconditional"),
            ) as req:
                result = model.generate_samples_from_batch(*positional, **kwargs, c3ache_request=req)
            summary = dict(req.stats, reason=req.reason)
        else:
            with ctrl if ctrl else nullcontext():
                result = model.generate_samples_from_batch(*positional, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if ctrl:
            summary = ctrl.finish()
            if args.worker == "asi":
                assert (summary["dense_stack_count"], summary["sparse_stack_count"]) == (1, 7)
                assert ctrl.plan["execution_mask"].sum(-1).tolist() == [184] * 8
                selection_audit[mode] = {"mask": ctrl.plan["execution_mask"].detach().cpu().clone()}
            if args.worker == "toca":
                selection_audit[mode] = dict(
                    scores={str(k): v.detach().cpu().clone() for k, v in ctrl.scores.items()},
                    indices={str(k): v.detach().cpu().clone() for k, v in ctrl.indices.items()},
                )
        if strategy and args.worker == "worldcache":
            summary = model._last_worldcache_report
            assert (summary["full_forwards"], summary["cache_forwards"]) == (6, 2)
        if strategy and cache is not None:
            expected = (8, 0) if chunk % 2 == 0 else (4, 4)
            assert (summary["dense_forwards"], summary["cache_hits"]) == expected, summary
        values = {k: result[k][0].detach().cpu().clone() for k in ("action", "vision")}
        assert all(torch.isfinite(v).all() for v in values.values()), "NaN/Inf"
        return values, elapsed, summary

    report = dict(
        source=str(TREES[args.worker]),
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        python=sys.executable,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        input=str(capture_path),
        input_sha256=input_hash,
        kwargs=capture["kwargs"],
        config=config,
        compile_dynamic=cfg.compile_dynamic,
        requested_compile=args.worker != "dense",
        requested_cuda_graph=args.worker != "dense",
        pad_for_cuda_graphs=False,
        asi_main_lse=args.worker == "asi",
        velocity_cache=False,
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_status=subprocess.check_output(["git", "status", "--short"], text=True).splitlines(),
        boundary="CUDA synchronized generation plus controller context; excludes summary, validation, CPU copies, VAE decode, RPC",
        warmups=args.warmups,
        repeats=args.repeats,
    )
    report["source_sha256"] = {
        str(p.relative_to(TREES[args.worker])): hashlib.sha256(p.read_bytes()).hexdigest()
        for name in (
            "inference/toca_future.py",
            "inference/toca_compiled.py",
            "inference/toca_joint_attention.py",
            "inference/worldcache.py",
            "inference/c3ache.py",
            "inference/edge_core_stable.py",
            "inference/edge_core_stable_fast.py",
            "inference/asi_main_lse.py",
            "inference/asi_existing_ablation.py",
            "model/generator/mot/attention.py",
            "model/generator/mot/inference_text_kv_memory.py",
            "model/generator/omni_mot_model.py",
            "model/generator/mot/unified_mot.py",
            "model/generator/mot/cosmos3_vfm_network.py",
        )
        if (p := TREES[args.worker] / "cosmos_framework" / name).exists()
    }
    save(out / "manifest.json", report)
    try:
        with torch.inference_mode():
            layout_records = []

            def layout_hook(module, positional, kwargs):
                pack = kwargs.get("input", positional[0] if positional else None)
                if isinstance(pack, dict):
                    layout_records.append({key: pack.get(key) for key in ("_num_full_tokens", "_num_causal_tokens")})

            handle = eager_layers[0].register_forward_pre_hook(layout_hook, with_kwargs=True)
            try:
                dense, _, _ = run("dense_eager")
            finally:
                handle.remove()
            report["dense_layout_records"] = layout_records
            if args.worker == "dense":
                torch.save(dense, out / "reference.pt")
                seconds = []
                for i in range(args.warmups + args.repeats):
                    value, elapsed, _ = run("dense_eager")
                    for k in value:
                        torch.testing.assert_close(value[k], dense[k], rtol=0, atol=0)
                    if i >= args.warmups:
                        seconds.append(dict(repeat=i - args.warmups, mode="dense_eager", seconds=elapsed))
                    print(f"[chunk] dense {i + 1}/{args.warmups + args.repeats} {elapsed:.6f}s", flush=True)
                x = [row["seconds"] for row in seconds]
                report.update(
                    samples=seconds,
                    finite=True,
                    timing={
                        "dense_eager": dict(
                            mean_s=float(np.mean(x)),
                            median_s=float(np.median(x)),
                            p90_s=float(np.quantile(x, 0.9)),
                            n=len(x),
                        )
                    },
                )
                save(out / "results.json", report)
                return
            reference = torch.load(args.output / "dense/reference.pt", map_location="cpu", weights_only=False)
            for k in dense:
                torch.testing.assert_close(dense[k], reference[k], rtol=0, atol=0)
            report["dense_matches_pristine_checkout_exact"] = True
            eager, _, eager_summary = run("strategy_eager")
            if args.adapted and args.worker in ("toca", "worldcache"):
                adapted, _, _ = run("strategy_adapted_eager")
                for k in adapted:
                    torch.testing.assert_close(adapted[k], eager[k], rtol=0, atol=0)
                if args.worker == "toca":
                    for field in ("scores", "indices"):
                        for key, value in selection_audit["strategy_eager"][field].items():
                            torch.testing.assert_close(
                                value, selection_audit["strategy_adapted_eager"][field][key], rtol=0, atol=0
                            )
                report["adapter_eager_exact"] = True
            if cache:
                cache.episodes.clear()
            modes = ["dense_eager", "refresh_graph", "hit_graph"] if cache else ["dense_eager", "strategy_graph"]
            rows, refs, last, masks = [], {}, {}, {}
            for i in range(args.warmups + args.repeats):
                # Keep refresh immediately followed by hit; alternate pair order vs dense.
                order = modes if i % 2 == 0 else (modes[1:] + modes[:1])
                for mode in order:
                    chunk = 2 * i + (mode == "hit_graph")
                    values, elapsed, summary = run(mode, chunk)
                    if i >= args.warmups:
                        for k in values:
                            torch.testing.assert_close(values[k], refs[mode][k], rtol=1e-5, atol=1e-5)
                        if args.worker == "asi" and mode == "strategy_graph":
                            assert torch.equal(selection_audit[mode]["mask"], masks[mode]), (
                                "Mask changed on repeated input"
                            )
                        rows.append(dict(repeat=i - args.warmups, mode=mode, seconds=elapsed))
                    refs[mode], last[mode] = values, summary
                    if args.worker == "asi" and mode == "strategy_graph":
                        masks[mode] = selection_audit[mode]["mask"]
                    print(
                        f"[chunk] {args.worker} {i + 1}/{args.warmups + args.repeats} {mode} {elapsed:.6f}s", flush=True
                    )
            report["samples"] = rows
            report["timing"] = {}
            for mode in modes:
                x = [r["seconds"] for r in rows if r["mode"] == mode]
                report["timing"][mode] = dict(
                    mean_s=float(np.mean(x)), median_s=float(np.median(x)), p90_s=float(np.quantile(x, 0.9)), n=len(x)
                )
            if cache:
                x = [
                    sum(r["seconds"] for r in rows if r["repeat"] == i and r["mode"] in ("refresh_graph", "hit_graph"))
                    / 2
                    for i in range(args.repeats)
                ]
                report["timing"]["amortized_graph"] = dict(
                    mean_s=float(np.mean(x)), median_s=float(np.median(x)), p90_s=float(np.quantile(x, 0.9)), n=len(x)
                )
            report["last_controller"] = last
            report["graph_vs_eager"] = {
                k: metric(refs["refresh_graph" if cache else "strategy_graph"][k], eager[k]) for k in eager
            }
            if args.worker == "toca":
                original, compiled = selection_audit["strategy_eager"], selection_audit["strategy_graph"]
                report["toca_score_vs_eager"] = metric(
                    torch.cat(list(compiled["scores"].values())), torch.cat(list(original["scores"].values()))
                )
                report["toca_selected_membership"] = {
                    key: dict(
                        count=len(value),
                        same_order=bool(torch.equal(value, original["indices"][key])),
                        overlap_fraction=float(torch.isin(value, original["indices"][key]).float().mean()),
                    )
                    for key, value in compiled["indices"].items()
                }
            if args.worker == "asi":
                report["compiled_vs_eager_mask_xor"] = int(
                    (selection_audit["strategy_graph"]["mask"] != selection_audit["strategy_eager"]["mask"]).sum()
                )
            report["strategy_vs_dense"] = {
                mode: {k: metric(value[k], dense[k]) for k in value}
                for mode, value in refs.items()
                if mode != "dense_eager"
            }
            torch.save(refs, out / "outputs.pt")
            report["finite"] = True
            save(out / "results.json", report)
            # Separate untimed runtime audit: a requested graph flag is not evidence of replay.
            audit = {}
            for j, mode in enumerate(modes[1:]):
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                ) as prof:
                    run(mode, 2 * (args.warmups + args.repeats) + (mode == "hit_graph"))
                events = {
                    e.key: e.count
                    for e in prof.key_averages()
                    if "graph" in e.key.lower() or "compiled" in e.key.lower()
                }
                audit[mode] = events
                graph_launches = sum(n for key, n in events.items() if "cudaGraphLaunch" in key)
                assert graph_launches > 0, f"No Graph replay observed: {mode}"
            report["runtime_audit"] = audit
            if cache:
                cache.episodes.clear()
            changed, _, _ = run("refresh_graph" if cache else "strategy_graph", 0, seed_offset=1)
            assert not torch.equal(changed["vision"], refs["refresh_graph" if cache else "strategy_graph"]["vision"]), (
                "Stale output"
            )
            report["changed_seed_gate"] = True
            if args.worker in ("toca", "worldcache"):
                changed_eager, _, _ = run("strategy_eager", 0, seed_offset=1)
                report["changed_seed_graph_vs_eager"] = {k: metric(changed[k], changed_eager[k]) for k in changed}
            save(out / "results.json", report)
            print("[result] " + json.dumps(report["timing"]), flush=True)
    except Exception:
        report["error"] = traceback.format_exc()
        save(out / "failure.json", report)
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--capture", type=Path, default=CAPTURE)
    p.add_argument("--worker", choices=list(TREES))
    p.add_argument("--modes", nargs="+", default=list(TREES), choices=list(TREES))
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--adapted", action="store_true", default=True)
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.adapted:
        TREES.update(toca=ROOT / "worktrees/toca-future", worldcache=ROOT / "worktrees/worldcache")
    if args.warmups < 5 or args.repeats < 1:
        p.error("Require >=5 warmups and >=1 formal repeat")
    if args.worker:
        return worker(args)
    args.output.mkdir(parents=True, exist_ok=args.modes[0] != "dense")
    if args.modes[0] != "dense" and not (args.output / "dense/reference.pt").exists():
        p.error("Run pristine Dense first")
    for mode in args.modes:
        env = dict(
            os.environ,
            PYTHONPATH=f"{TREES[mode]}:{ROOT}/cosmos-edge-overlay",
            COSMOS_TRAINING="0",
            LD_LIBRARY_PATH="",
            HF_HOME="/root/cosmos3/cosmos/checkpoints/hf_home",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            CUDA_VISIBLE_DEVICES="0",
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
            TORCHINDUCTOR_CACHE_DIR=str(ROOT / "runtime/inductor_cache"),
        )
        cmd = [
            str(ASI / ".venv/bin/python"),
            str(Path(__file__).resolve()),
            "--worker",
            mode,
            "--output",
            str(args.output),
            "--capture",
            str(args.capture.resolve()),
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
        ]
        if args.adapted:
            cmd.append("--adapted")
        print("[start] " + mode, flush=True)
        with (args.output / f"{mode}.log").open("x") as f:
            result = subprocess.run(cmd, cwd=TREES[mode], env=env, stdout=f, stderr=subprocess.STDOUT)
        print(f"[end] {mode} exit={result.returncode}", flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
