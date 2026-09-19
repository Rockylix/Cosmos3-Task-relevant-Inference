"""Alternating Dense eager / frozen SpecPrune eager / compiled decoder + Graph.

Times include native preparation, online scoring/checks, selection, true sparse
layers, early-exit head and full UniPC. No paired-reference call, file IO, RPC or
VAE decode inside timing. Existing adapter CPU diagnostics are NOT subtracted.
"""

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.inference.specprune_exit_compile import compile_exit_layers
from cosmos_framework.inference.specprune_exit_metrics import future_vision
from cosmos_framework.inference.specprune_future import tensor_metrics
from cosmos_framework.inference.specprune_observation_exit import SpecPruneObservationExit
from cosmos_framework.scripts.action_policy_server_specprune import SpecPrunePolicyService, SpecPruneServerArgs


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def cost(rows, uc, uu, *, sparse):
    # Same logical matmul boundary as the other Edge baselines. MAC=2 FLOPs.
    d, q, kv, intermediate = 2048, 2048, 1024, 9216

    def gen(n, u):
        return 2 * n * d * (2 * q + 2 * kv) + 4 * n * d * intermediate + 4 * n * (n + u) * q

    def und(u):
        return 2 * u * d * (2 * q + 2 * kv) + 4 * u * d * intermediate + 4 * (u * (u + 1) // 2) * q

    g = sum(gen(r["gen_tokens"], uc if r["branch"] == "conditional" else uu) for r in rows)
    # Native Dense caches UND per branch; frozen adapter really recomputes it each step.
    u = 28 * (und(uc) + und(uu)) * (4 if sparse else 1)
    scoring = 0
    if sparse:
        for r in rows:
            if r["step"] == 0 and r["branch"] == "conditional" and r["block"] in (0, 1, 13, 14, 19, 24, 27):
                nq = 340 if r["block"] in (0, 1, 13, 27) else 32
                scoring += 4 * nq * (r["gen_tokens"] + uc) * q  # explicit QK and validation AV
    return dict(gen=g, und=u, scoring=scoring, total=g + u + scoring)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--warmups", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--eager-only", action="store_true")
    args = ap.parse_args()
    if args.warmups < 5 or args.repeats < 1:
        ap.error("Need >=5 warmups and >=1 repeats")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.accumulated_recompile_limit = 256
    service = SpecPrunePolicyService(
        SpecPruneServerArgs(
            checkpoint_path=args.checkpoint,
            specprune=False,
            num_steps=4,
            shift=5,
            guidance=3,
            seed=0,
            deterministic_seed=False,
            format_prompt_as_json=True,
            guardrails=False,
            decode_video=False,
            output_dir=args.output / "model_output",
            experiment_overrides=[
                f"model.config.tokenizer.vae_path={args.vae}",
                "model.config.tokenizer.object_store_credential_path_pretrained=",
                "model.config.tokenizer.bucket_name=",
            ],
        )
    )
    model = service.model
    record = torch.load(args.capture, map_location="cpu", weights_only=False)
    kwargs = {k: record["kwargs"][k] for k in ("seed", "num_steps", "shift", "guidance")}
    assert (kwargs["num_steps"], kwargs["shift"], kwargs["guidance"]) == (4, 5, 3)
    layers = model.net.language_model.model.layers
    # Confirm the dimensions used in logical FLOPs against loaded weights.
    assert layers[0].mlp_moe_gen.up_proj.weight.shape == (9216, 2048)
    assert layers[0].self_attn.q_proj_moe_gen.weight.shape == (2048, 2048)
    assert layers[0].self_attn.k_proj_moe_gen.weight.shape == (1024, 2048)
    eager = SpecPruneObservationExit(model)
    compiled = (
        None
        if args.eager_only
        else SpecPruneObservationExit(model, compiled_layers=compile_exit_layers(model, cuda_graphs=True))
    )
    layouts = []

    def hook(module, positional, named):
        pack = named.get("input", positional[0] if positional else None)
        if isinstance(pack, dict):
            layouts.append([pack["_num_full_tokens"], pack["_num_causal_tokens"]])

    with torch.inference_mode():
        h = layers[0].register_forward_pre_hook(hook, with_kwargs=True)
        try:
            dense = model.generate_samples_from_batch(copy.deepcopy(record["args"][0]), **kwargs)
        finally:
            h.remove()
        assert len(layouts) == 8 and all(n == 3093 for n, u in layouts)
        uc, uu = layouts[0][1], layouts[1][1]
        full = eager.generate(copy.deepcopy(record["args"][0]), **kwargs, force_full=True)
        assert all(
            full[k][0].shape == dense[k][0].shape and torch.equal(full[k][0], dense[k][0]) for k in ("action", "vision")
        )
        # Identical fixed input twice builds controlled history, NOT real previous c1/c2 observations.
        eager.reset()
        for i in range(2):
            eager.generate(copy.deepcopy(record["args"][0]), **kwargs)
        history = copy.deepcopy({k: getattr(eager.plan, k) for k in ("previous_rgb", "previous_global", "confidence")})
        modes = ["dense", "first_eager", "history_eager"]
        if compiled is not None:
            modes += ["first_graph", "history_graph"]
        last, refs, timings, token_rows = {}, {}, [], {}

        def run(mode, seed_delta=0):
            batch = copy.deepcopy(record["args"][0])
            kw = copy.deepcopy(kwargs)
            kw["seed"] = [kw["seed"][0] + seed_delta]
            random.seed(kw["seed"][0])
            np.random.seed(kw["seed"][0])
            torch.manual_seed(kw["seed"][0])
            torch.cuda.manual_seed_all(kw["seed"][0])
            adapter = None if mode == "dense" else compiled if mode.endswith("graph") else eager
            if adapter:
                adapter.reset()
                if mode.startswith("history"):
                    for k, v in copy.deepcopy(history).items():
                        setattr(adapter.plan, k, v)
                    adapter.chunk = 2
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = (model.generate_samples_from_batch if adapter is None else adapter.generate)(batch, **kw)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            values = {k: result[k][0].detach().cpu().clone() for k in ("action", "vision")}
            assert all(torch.isfinite(v).all() for v in values.values())
            if adapter:
                last[mode] = copy.deepcopy(adapter.last_info)
                token_rows[mode] = copy.deepcopy(adapter.last_token_rows)
                values["masks"] = {str(k): v.cpu().clone() for k, v in adapter.plan.masks.items()}
            return values, seconds

        for i in range(args.warmups + args.repeats):
            order = modes if i % 2 == 0 else modes[::-1]
            for mode in order:
                values, seconds = run(mode)
                if i >= args.warmups:
                    if mode in refs:
                        assert torch.equal(refs[mode]["action"], values["action"]) and torch.equal(
                            refs[mode]["vision"], values["vision"]
                        ), "Same-mode output changed"
                    timings.append(dict(repeat=i - args.warmups, mode=mode, seconds=seconds))
                    refs[mode] = values
                print(f"BENCH {i + 1}/{args.warmups + args.repeats} {mode} {seconds:.6f}s", flush=True)
        stats = {}
        dense_rows = [
            dict(step=s, branch=b, block=l, gen_tokens=3093)
            for s in range(4)
            for b in ("conditional", "unconditional")
            for l in range(28)
        ]
        dense_cost = cost(dense_rows, uc, uu, sparse=False)
        for mode in modes:
            x = [r["seconds"] for r in timings if r["mode"] == mode]
            logical = cost(token_rows.get(mode, dense_rows), uc, uu, sparse=mode != "dense")
            stats[mode] = dict(
                mean_s=float(np.mean(x)),
                median_s=float(np.median(x)),
                p90_s=float(np.percentile(x, 90)),
                n=len(x),
                logical_flops=logical,
                flops_percent=100 * logical["total"] / dense_cost["total"],
            )
        for row in stats.values():
            row["speedup"] = stats["dense"]["median_s"] / row["median_s"]
        comparisons = {}
        for mode in modes[1:]:
            comparisons[mode] = {
                "vs_dense_action": tensor_metrics(refs[mode]["action"][1:, :8], refs["dense"]["action"][1:, :8]),
                "vs_dense_future": tensor_metrics(
                    future_vision(refs[mode]["vision"]), future_vision(refs["dense"]["vision"])
                ),
            }
            if mode.endswith("graph"):
                base = mode.replace("graph", "eager")
                comparisons[mode].update(
                    vs_eager_action=tensor_metrics(refs[mode]["action"], refs[base]["action"]),
                    vs_eager_future=tensor_metrics(
                        future_vision(refs[mode]["vision"]), future_vision(refs[base]["vision"])
                    ),
                    mask_xor={k: int((v != refs[base]["masks"][k]).sum()) for k, v in refs[mode]["masks"].items()},
                )
        root = Path(__file__).resolve().parents[1]
        sources = list((root / "cosmos_framework/inference").glob("*specprune*.py")) + [
            root / "tools/benchmark_specprune_exit.py",
            root / "cosmos_framework/inference/future_instruction_attention.py",
        ]
        report = dict(
            timing=stats,
            comparisons=comparisons,
            kwargs=kwargs,
            full_keep_exact=True,
            layouts=layouts,
            input_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
            last_controller=last,
            boundary=__doc__,
            history="two identical recorded observations; reset frozen history before every measurement",
            source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            flops_scope="Logical decoder Q/K/V/O, QK/AV, MLP + explicit scoring QK/AV; excludes VAE, heads, redundant _view input projections, norms, softmax, indexing, transfers, UniPC. Not full-pipeline FLOPs.",
            compiler="Non-capture layers only; fullgraph/dynamic/reduce-overhead; eager selection and 7 capture layers retained",
        )
        save(args.output / "timing_samples.json", timings)
        save(args.output / "results.json", report)
        torch.save(refs, args.output / "outputs.pt")
        for mode in [m for m in modes if m.endswith("graph")]:
            changed, _ = run(mode, seed_delta=1)
            assert not torch.equal(changed["action"], refs[mode]["action"]), "Stale graph output suspected"
            run(mode)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                run(mode)
            events = {
                e.key: e.count for e in prof.key_averages() if "graph" in e.key.lower() or "compiled" in e.key.lower()
            }
            report.setdefault("graph_audit", {})[mode] = events
            save(args.output / "results.json", report)
            assert sum(n for k, n in events.items() if "cudaGraphLaunch" in k) > 0, "No verified graph replay"
        report["complete"] = True
        save(args.output / "results.json", report)
        print(json.dumps(stats, indent=2), flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
