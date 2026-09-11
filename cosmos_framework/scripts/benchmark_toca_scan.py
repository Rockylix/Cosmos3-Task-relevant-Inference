"""Small-output paired timing and decoded-future fidelity for the ToCa grid."""

from cosmos_framework.inference.common.init import init_script

init_script()

import argparse
import copy
import json
import random
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.inference.future_fidelity_metrics import rgb_frame_metrics, tensor_metrics, to_rgb01
from cosmos_framework.inference.toca_future import ToCaFutureConfig, ToCaFutureController
from cosmos_framework.inference.toca_scan_plan import ROBOLAB, ROOT, TASKS, VAE, configurations
from cosmos_framework.scripts.action_policy_server_robolab import RobolabServerArgs
from cosmos_framework.scripts.paired_future_fidelity import EagerService


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_model(out):
    return EagerService(
        RobolabServerArgs(
            checkpoint_path=str(ROBOLAB / "Cosmos3-Edge-Policy-DROID"),
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
    ).model


def generate(model, capture, values, *, audit=False):
    args, kwargs = copy.deepcopy(capture["args"]), copy.deepcopy(capture["kwargs"])
    seed = kwargs["seed"][0]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = ToCaFutureConfig(**{**values, "full_steps": tuple(values["full_steps"])}) if values is not None else None
    traces, handles = {}, []
    if audit:
        for block, layer in enumerate(model.net.language_model.model.layers):
            modules = {name: getattr(layer.self_attn, f"{name}_proj_moe_gen") for name in ("q", "k", "v", "o")}
            modules["mlp"] = layer.mlp_moe_gen
            for name, module in modules.items():
                key = (block, name)
                traces[key] = []
                handles.append(module.register_forward_pre_hook(lambda m, x, key=key: traces[key].append(len(x[0]))))
    torch.cuda.synchronize()
    start = time.perf_counter()
    controller = ToCaFutureController(model.net, config) if config else None
    try:
        with controller if controller is not None else nullcontext():
            result = model.generate_samples_from_batch(*args, **kwargs)
        summary = controller.finish() if controller else {}
    finally:
        for handle in handles:
            handle.remove()
    for key in ("action", "vision"):
        if not torch.isfinite(result[key][0]).all().item():
            raise FloatingPointError(f"Nonfinite {key}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if audit:
        for block in range(28):
            for name, field in (
                ("q", "q_rows"),
                ("k", "kv_rows"),
                ("v", "kv_rows"),
                ("o", "o_rows"),
                ("mlp", "mlp_rows"),
            ):
                expected = [r[field] for r in controller.records if r["block"] == block] if controller else [3093] * 8
                assert traces[block, name] == expected, (block, name, traces[block, name], expected)
    output = {key: result[key][0].detach().cpu() for key in ("action", "vision")}
    return output, elapsed, {k: v for k, v in summary.items() if k != "layout"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.warmups < 2 or args.repeats < 3:
        raise ValueError("Insufficient warmups/repeats")
    run = args.run_dir.resolve()
    out = run / ("validation" if args.validate_only else "paired")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model = load_model(out)
    if model.tokenizer_vision_gen._keep_decoder_cache:
        raise RuntimeError("Paired decode must not reuse decoder state")
    configs = configurations()
    modes = tuple(configs)
    with torch.inference_mode():
        if args.validate_only:
            path = (
                ROOT
                / "experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2/dense/server/captures/request_000002/sample.pt"
            )
            cap = torch.load(path, map_location="cpu", weights_only=False)
            checks = []
            for mode in modes:
                values, _, summary = generate(model, cap, configs[mode], audit=True)
                if mode == "dense":
                    for key in values:
                        torch.testing.assert_close(values[key], cap["outputs"][key], rtol=1e-4, atol=1e-4)
                if mode == "dcdc_r0p25_b0_shared":
                    previous = torch.load(
                        ROOT / "experiments/toca_schedule_dccc_banana_c3_v1/toca_dcdc_outputs.pt",
                        map_location="cpu",
                        weights_only=False,
                    )
                    for key in values:
                        assert torch.equal(values[key], previous[key]), ("Default regression", key)
                checks.append({"mode": mode, "actual_module_rows_pass": True, "finite": True, **summary})
                print(f"[gate] {len(checks)}/25 {mode} PASS", flush=True)
            write_json(
                out / "summary.json", {"status": "complete", "checks": checks, "default_joint_regression_exact": True}
            )
        else:
            for task_index, task in enumerate(TASKS):
                case_file = out / f"{task}.json"
                if case_file.exists():
                    continue
                cap = torch.load(run / "_temporary_inputs" / f"{task}.pt", map_location="cpu", weights_only=False)
                # Dense + one candidate are decoded at a time; never persist RGB or tensors.
                dense, _, _ = generate(model, cap, None)
                for key in dense:
                    torch.testing.assert_close(dense[key], cap["outputs"][key], rtol=1e-4, atol=1e-4)
                dense_rgb = to_rgb01(model.decode(dense["vision"].cuda()).float().cpu().numpy())
                assert len(dense_rgb) == 33
                fidelity = {
                    "dense": {
                        "rgb_cosine": 1.0,
                        "rgb_relative_l2": 0.0,
                        "psnr_db": None,
                        "ssim": 1.0,
                        "psnr_infinite": True,
                    }
                }
                for mode in modes[1:]:
                    result, _, _ = generate(model, cap, configs[mode])
                    rgb = to_rgb01(model.decode(result["vision"].cuda()).float().cpu().numpy())
                    metrics = tensor_metrics(dense_rgb[1:], rgb[1:])
                    per_frame = [rgb_frame_metrics(dense_rgb[f], rgb[f]) for f in range(1, 33)]
                    psnr = [x["psnr_db"] for x in per_frame if x["psnr_db"] is not None]
                    fidelity[mode] = {
                        "rgb_cosine": metrics["cosine"],
                        "rgb_relative_l2": metrics["relative_l2"],
                        "psnr_db": statistics.mean(psnr) if len(psnr) == 32 else None,
                        "psnr_infinite": len(psnr) < 32,
                        "ssim": statistics.mean(x["ssim"] for x in per_frame),
                    }
                    del result, rgb
                del dense_rgb, dense
                for _ in range(args.warmups):
                    for mode in modes:
                        generate(model, cap, configs[mode])
                times = {mode: [] for mode in modes}
                for repeat in range(args.repeats):
                    offset = (repeat + task_index) % len(modes)
                    for mode in modes[offset:] + modes[:offset]:
                        _, elapsed, _ = generate(model, cap, configs[mode])
                        times[mode].append(elapsed)
                    if repeat % 5 == 4:
                        print(
                            f"[paired] task={task_index + 1}/10 {task} timing={repeat + 1}/{args.repeats}", flush=True
                        )
                stats = {
                    mode: {
                        "n": len(x),
                        "mean_s": statistics.mean(x),
                        "median_s": statistics.median(x),
                        "p90_s": float(np.quantile(x, 0.9)),
                    }
                    for mode, x in times.items()
                }
                for mode, stat in stats.items():
                    stat["speedup_vs_dense"] = stats["dense"]["median_s"] / stat["median_s"]
                write_json(
                    case_file,
                    {
                        "task": task,
                        "chunk": cap["metadata"]["prompt_chunk"],
                        "seed": cap["kwargs"]["seed"],
                        "timing": stats,
                        "fidelity": fidelity,
                        "all_finite": True,
                    },
                )
                print(f"[paired] completed {task_index + 1}/10 {task}", flush=True)
            write_json(
                out / "summary.json",
                {"status": "complete", "tasks": TASKS, "warmups": args.warmups, "repeats": args.repeats},
            )
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
