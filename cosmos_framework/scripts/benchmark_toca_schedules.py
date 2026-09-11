"""Paired warm Edge generation timing: Dense versus joint ToCa DCDC/DCCC."""

from cosmos_framework.inference.common.init import init_script

init_script()

import argparse
import copy
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.inference.future_fidelity_metrics import tensor_metrics
from cosmos_framework.inference.toca_future import ToCaFutureConfig, ToCaFutureController
from cosmos_framework.scripts.action_policy_server_robolab import RobolabServerArgs
from cosmos_framework.scripts.paired_future_fidelity import EagerService, write_csv, write_json

CONFIGS = {
    "dense": None,
    "toca_dcdc": ToCaFutureConfig(full_steps=(0, 2), period=2, attention_backend="joint"),
    "toca_dccc": ToCaFutureConfig(full_steps=(0,), period=4, attention_backend="joint"),
}


def generate(model, capture, mode):
    args, kwargs = copy.deepcopy(capture["args"]), copy.deepcopy(capture["kwargs"])
    seed = kwargs["seed"][0]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Native generate constructs fresh noise, UND KV state, and solver history.
    # Each run also has a fresh ToCa controller; no cross-run cache reuse.
    torch.cuda.synchronize()
    started = time.perf_counter()
    config = CONFIGS[mode]
    controller = ToCaFutureController(model.net, config) if config is not None else None
    with controller if controller is not None else nullcontext():
        output = model.generate_samples_from_batch(*args, **kwargs)
    summary = controller.finish() if controller is not None else {}
    for name in ("action", "vision"):
        if not torch.isfinite(output[name][0]).all().item():
            raise FloatingPointError(f"Nonfinite {mode}/{name}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    samples = {name: output[name][0].detach().cpu() for name in ("action", "vision")}
    records = controller.records if controller is not None else []
    return samples, elapsed, summary, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.repeats < 3 or args.warmups < 2:
        raise ValueError("At least two warmups and three repeats required")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    capture_path = args.capture.resolve()
    capture = torch.load(capture_path, map_location="cpu", weights_only=False)
    assert capture["metadata"]["prompt_chunk"] == 3
    print("[capture]", capture_path, capture["metadata"], capture["kwargs"], flush=True)
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[2]
    checkpoint = Path("/root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID")
    vae = Path(
        "/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/"
        "921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
    )
    assert checkpoint.is_dir() and vae.is_file()
    manifest = {
        "capture": str(capture_path),
        "metadata": capture["metadata"],
        "configs": {mode: asdict(c) if c is not None else None for mode, c in CONFIGS.items()},
        "python": sys.executable,
        "pythonpath": os.environ.get("PYTHONPATH"),
        "source": str(root),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "cosmos_framework/inference/toca_future.py",
                "cosmos_framework/inference/toca_joint_attention.py",
                "cosmos_framework/scripts/benchmark_toca_schedules.py",
            )
        },
        "checkpoint": str(checkpoint),
        "vae": str(vae),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "compile": False,
        "cuda_graphs": False,
        "warmups_per_mode": args.warmups,
        "repeats_per_mode": args.repeats,
        "timing_boundary": "synchronized generation including controller and finite checks; excluding deepcopy, RNG reset, CPU copies, decode and IO",
    }
    write_json(out / "manifest.json", manifest)
    model = EagerService(
        RobolabServerArgs(
            checkpoint_path=str(checkpoint),
            seed=0,
            deterministic_seed=False,
            guidance=3,
            num_steps=4,
            shift=5,
            format_prompt_as_json=True,
            guardrails=False,
            output_dir=out / "model_output",
            experiment_overrides=[
                f"model.config.tokenizer.vae_path={vae}",
                "model.config.tokenizer.object_store_credential_path_pretrained=",
                "model.config.tokenizer.bucket_name=",
            ],
        )
    ).model
    outputs, checks, rows = {}, {}, []
    modes = tuple(CONFIGS)
    with torch.inference_mode():
        for mode in modes:
            result, _, summary, records = generate(model, capture, mode)
            outputs[mode] = result
            if mode == "dense":
                for name in result:
                    torch.testing.assert_close(result[name], capture["outputs"][name], rtol=1e-4, atol=1e-4)
            else:
                for record in records:
                    expected = "full" if record["step"] in CONFIGS[mode].full_steps else "cached"
                    assert record["mode"] == expected
                    assert record["kv_rows"] == 3093
                    assert record["q_rows"] == (3093 if expected == "full" else 373)
                assert len(records) == 224
            checks[mode] = summary
            rows.extend({"strategy": mode, **r} for r in records)
            torch.save(result, out / f"{mode}_outputs.pt")
            print(f"[validation] {mode} finite and schedule passed", flush=True)
        write_json(out / "checks.json", checks)
        write_csv(out / "module_compute.csv", rows)
        metrics = []
        for mode in modes[1:]:
            for name in ("action", "vision"):
                values = [outputs[m][name].float().numpy() for m in ("dense", mode)]
                values = [x[1:] if name == "action" else x[:, :, 1:] for x in values]
                metrics.append({"mode": mode, "modality": name, **tensor_metrics(*values)})
        write_csv(out / "fidelity.csv", metrics)
        for repeat in range(args.warmups):
            for mode in modes:
                generate(model, capture, mode)
            print(f"[warmup] {repeat + 1}/{args.warmups}", flush=True)
        timing = []
        for repeat in range(args.repeats):
            order = modes[repeat % 3 :] + modes[: repeat % 3]
            for mode in order:
                result, elapsed, _, _ = generate(model, capture, mode)
                for name in result:
                    assert torch.equal(result[name], outputs[mode][name]), ("Repeat output changed", repeat, mode, name)
                timing.append({"repeat": repeat, "mode": mode, "generation_s": elapsed})
            if repeat % 5 == 4:
                write_csv(out / "timing.csv", timing)
                print(f"[timing] {repeat + 1}/{args.repeats}", flush=True)
        stats = {}
        for mode in modes:
            values = [r["generation_s"] for r in timing if r["mode"] == mode]
            stats[mode] = {
                "n": len(values),
                "mean_s": statistics.mean(values),
                "median_s": statistics.median(values),
                "p90_s": float(np.quantile(values, 0.9)),
            }
        for row in stats.values():
            row["speedup_vs_dense"] = stats["dense"]["median_s"] / row["median_s"]
        summary = {
            "status": "complete",
            "timing": stats,
            "dccc_speedup_vs_dcdc": stats["toca_dcdc"]["median_s"] / stats["toca_dccc"]["median_s"],
            "all_finite": True,
            "all_repeated_outputs_exact": True,
            "closed_loop_tested": False,
        }
        write_json(out / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
