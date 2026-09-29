"""Branch-native paired eager chunk benchmark; no model modifications/downloads."""

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/root/robolab")
HERE = Path(__file__).resolve()
PYTHON = ROOT / "cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python"
TREES = {
    "baseline": ROOT / "cosmos-framework-edge",
    "asi": ROOT / "cosmos-framework-edge-core80-stable104-action-weighted",
    "toca": ROOT / "worktrees/toca-future",
    "worldcache": ROOT / "worktrees/worldcache",
    "c3ache": ROOT / "worktrees/c3ache",
}
INPUT = (
    TREES["toca"]
    / "experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2/dense/server/captures/request_000002/sample.pt"
)
VAE = Path(
    "/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def worker(args):
    from cosmos_framework.inference.common.init import init_script

    init_script()
    import random
    from contextlib import nullcontext

    import numpy as np
    import torch

    import cosmos_framework
    from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService, RobolabServerArgs

    assert Path(cosmos_framework.__path__[0]).resolve() == TREES[args.worker] / "cosmos_framework"
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    out = args.output / args.worker
    out.mkdir(parents=True, exist_ok=False)

    class EagerService(RobolabPolicyService):
        def _build_setup_args(self, settings):
            return (
                super()
                ._build_setup_args(settings)
                .model_copy(update={"use_torch_compile": False, "use_cuda_graphs": False})
            )

    settings = RobolabServerArgs(
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
    service = EagerService(settings)
    model = service.model
    capture = torch.load(INPUT, map_location="cpu", weights_only=False)
    assert capture["kwargs"] == {"guidance": 3.0, "seed": [1097657232], "num_steps": 4, "shift": 5.0}
    config = None
    if args.worker == "asi":
        from cosmos_framework.scripts.robolab_version1 import Version1Controller
    elif args.worker == "toca":
        from cosmos_framework.inference.toca_future import ToCaFutureConfig, ToCaFutureController

        config = json.loads((TREES["toca"] / "configs/toca_baseline.json").read_text())["config"]
        config["full_steps"] = tuple(config["full_steps"])
        toca_config = ToCaFutureConfig(**config)
    elif args.worker == "worldcache":
        from cosmos_framework.inference.worldcache import WorldCacheConfig

        config = json.loads((TREES["worldcache"] / "configs/worldcache_baseline.json").read_text())["config"]
        wc_config = WorldCacheConfig(**config)
    elif args.worker == "c3ache":
        from cosmos_framework.inference.c3ache import C3acheCache, C3acheConfig

        config = json.loads((TREES["c3ache"] / "configs/c3ache_baseline.json").read_text())["config"]
        cache = C3acheCache(C3acheConfig(**config))

    def metric(x, y):
        x, y = x.double().flatten(), y.double().flatten()
        return {
            "mse": float((x - y).square().mean()),
            "relative_l2": float((x - y).norm() / y.norm().clamp_min(1e-20)),
            "max_absolute_error": float((x - y).abs().max()),
            "cosine": float((x @ y) / (x.norm() * y.norm()).clamp_min(1e-20)),
        }

    def run(mode, chunk=0, audit=False):
        positional, kwargs = copy.deepcopy(capture["args"]), copy.deepcopy(capture["kwargs"])
        seed = kwargs["seed"][0]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        counts = {"layer0": 0, "net": 0}
        handles = []
        if audit:

            def count_layer(*_):
                counts["layer0"] += 1

            def count_net(*_):
                counts["net"] += 1

            handles = [
                model.net.register_forward_pre_hook(count_net),
                model.net.language_model.model.layers[0].register_forward_pre_hook(count_layer),
            ]
        torch.cuda.synchronize()
        start = time.perf_counter()
        controller, summary = None, {}
        try:
            if mode == "asi":
                controller = Version1Controller(torch=torch, net=model.net, guidance=3, num_steps=4)
            elif mode == "toca":
                controller = ToCaFutureController(model.net, toca_config)
            elif mode == "worldcache":
                kwargs["worldcache_config"] = wc_config
            if mode.startswith("c3ache"):
                obs = {"session_id": "paired-benchmark", "episode_id": "fixed-observation-replay", "chunk_id": chunk}
                with cache.request(
                    obs,
                    signature={
                        "prompt": capture["metadata"]["prompt"],
                        "steps": 4,
                        "guidance": 3,
                        "shift": 5,
                        "sampler": "unipc",
                    },
                    transformer=model.net.language_model.model,
                    net=model.net,
                    branches=("conditional", "unconditional"),
                ) as request:
                    result = model.generate_samples_from_batch(*positional, **kwargs, c3ache_request=request)
                summary = dict(request.stats, reason=request.reason, chunk_id=chunk)
            else:
                with controller if controller else nullcontext():
                    result = model.generate_samples_from_batch(*positional, **kwargs)
                if controller:
                    summary = controller.finish()
                if mode == "worldcache":
                    summary = model._last_worldcache_report
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        finally:
            for h in handles:
                h.remove()
        # Output validation, CPU copies, metric reductions and artifact I/O are OUTSIDE timing.
        values = {k: result[k][0].detach().cpu() for k in ("action", "vision")}
        for v in values.values():
            assert torch.isfinite(v).all().item(), (mode, "NaN/Inf")
        if mode == "c3ache_refresh":
            assert (summary["dense_forwards"], summary["cache_hits"]) == (8, 0), summary
        if mode == "c3ache_hit":
            assert (summary["dense_forwards"], summary["cache_hits"]) == (4, 4), summary
        if mode == "worldcache":
            assert (summary["full_forwards"], summary["cache_forwards"]) == (6, 2), summary
        return values, elapsed, summary, counts

    with torch.inference_mode():
        dense, _, _, count = run("dense", audit=True)
        assert count == {"layer0": 8, "net": 8}, count
        reference = {k: metric(dense[k], capture["outputs"][k]) for k in dense}
        hashes = {
            k: hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k, v in dense.items()
        }
        gates = {"dense_counts": count, "dense_vs_stored_dense": reference, "dense_hashes": hashes, "finite": True}
        if args.worker == "c3ache":
            for mode, chunk in [("c3ache_refresh", 0), ("c3ache_hit", 1)]:
                values, _, report, counts = run(mode, chunk, audit=True)
                assert counts["layer0"] == (8 if chunk == 0 else 4), counts
                gates[mode] = {
                    "counts": counts,
                    "controller": report,
                    "vs_dense": {k: metric(values[k], dense[k]) for k in dense},
                }
                if chunk == 0:
                    for k in dense:
                        torch.testing.assert_close(values[k], dense[k], rtol=0, atol=0)
            cache.episodes.clear()
        elif args.worker != "baseline":
            values, _, report, counts = run(args.worker, audit=True)
            gates[args.worker] = {
                "counts": counts,
                "controller": report,
                "vs_dense": {k: metric(values[k], dense[k]) for k in dense},
            }
        write_json(out / "gate.json", gates)
        print(f"[gate] {args.worker} PASS", flush=True)
        rows = []
        last_summary = {}
        for i in range(args.warmups + args.repeats):
            if args.worker == "c3ache":
                cycle = [("c3ache_refresh", 2 * i), ("c3ache_hit", 2 * i + 1)]
                order = [("dense", 0)] + cycle if i % 2 == 0 else cycle + [("dense", 0)]
            elif args.worker == "baseline":
                order = [("dense", 0)]
            else:
                modes = ["dense", args.worker] if i % 2 == 0 else [args.worker, "dense"]
                order = [(m, 0) for m in modes]
            for mode, chunk in order:
                _, elapsed, summary, _ = run(mode, chunk)
                last_summary[mode] = summary
                if i >= args.warmups:
                    rows.append({"repeat": i - args.warmups, "mode": mode, "seconds": elapsed})
            print(f"[timing] {args.worker} cycle={i + 1}/{args.warmups + args.repeats} last={elapsed:.6f}s", flush=True)
        timing = {}
        for mode in dict.fromkeys(r["mode"] for r in rows):
            samples = [r["seconds"] for r in rows if r["mode"] == mode]
            timing[mode] = {
                "n": len(samples),
                "mean_s": float(np.mean(samples)),
                "median_s": float(np.median(samples)),
                "p90_s": float(np.quantile(samples, 0.9)),
            }
        if args.worker == "c3ache":
            means = [
                sum(r["seconds"] for r in rows if r["repeat"] == i and r["mode"].startswith("c3ache")) / 2
                for i in range(args.repeats)
            ]
            timing["c3ache_period2_amortized"] = {
                "n": len(means),
                "mean_s": float(np.mean(means)),
                "median_s": float(np.median(means)),
                "p90_s": float(np.quantile(means, 0.9)),
            }
        for stat in timing.values():
            stat["speedup_vs_paired_dense"] = timing["dense"]["median_s"] / stat["median_s"]
        with (out / "timings.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["repeat", "mode", "seconds"])
            w.writeheader()
            w.writerows(rows)
        write_json(
            out / "summary.json",
            {
                "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
                "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "source": str(TREES[args.worker]),
                "python": sys.executable,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "compile": False,
                "cuda_graphs": False,
                "input": str(INPUT),
                "input_sha256": sha(INPUT),
                "metadata": capture["metadata"],
                "kwargs": capture["kwargs"],
                "config": config,
                "warmups_each": args.warmups,
                "timing": timing,
                "last_controller_reports": last_summary,
                "gates": gates,
                "c3ache_scope": "Repeated identical observation with advancing cache chunk IDs; timing only, not real cross-chunk accuracy.",
                "boundary": "generate_samples_from_batch plus controller lifecycle; CUDA synchronized; excludes validation/copies/decode/RPC",
            },
        )
        print("[result] " + json.dumps(timing), flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", choices=list(TREES))
    p.add_argument("--modes", nargs="+", default=list(TREES), choices=list(TREES))
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        env = dict(
            os.environ,
            COSMOS_TRAINING="0",
            LD_LIBRARY_PATH="",
            PYTHONPATH=f"{TREES[mode]}:{ROOT / 'cosmos-edge-overlay'}",
            HF_HOME="/root/cosmos3/cosmos/checkpoints/hf_home",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            CUDA_VISIBLE_DEVICES="0",
            NO_PROXY="127.0.0.1,localhost",
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
        )
        command = [
            str(PYTHON),
            str(HERE),
            "--worker",
            mode,
            "--output",
            str(args.output),
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
        ]
        print(f"[start] {mode}", flush=True)
        with (args.output / f"{mode}.log").open("x") as log:
            subprocess.run(command, cwd=TREES[mode], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"[done] {mode}", flush=True)


if __name__ == "__main__":
    main()
