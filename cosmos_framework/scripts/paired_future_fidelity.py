"""Dense/ASI/ToCa on fixed Dense chunk-3 captures, plus same-input timing."""

from cosmos_framework.inference.common.init import init_script

init_script()

import argparse
import copy
import csv
import json
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from cosmos_framework.inference.future_fidelity_metrics import rgb_frame_metrics, tensor_metrics, to_rgb01
from cosmos_framework.inference.toca_future import ToCaFutureController
from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService, RobolabServerArgs
from cosmos_framework.scripts.robolab_version1 import Version1Controller


class EagerService(RobolabPolicyService):
    def _build_setup_args(self, args):
        return super()._build_setup_args(args).model_copy(update={"use_torch_compile": False, "use_cuda_graphs": False})


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_one(model, capture, mode):
    args, kwargs = copy.deepcopy(capture["args"]), copy.deepcopy(capture["kwargs"])
    torch.cuda.synchronize()
    started = time.perf_counter()
    controller = None
    if mode == "asi":
        controller = Version1Controller(torch=torch, net=model.net, guidance=3, num_steps=4)
    elif mode == "toca_future":
        controller = ToCaFutureController(model.net)
    elif mode != "dense":
        raise ValueError(mode)
    with controller if controller is not None else nullcontext():
        output = model.generate_samples_from_batch(*args, **kwargs)
    summary = controller.finish() if controller is not None else {}
    for name in ("action", "vision"):
        if not torch.isfinite(output[name][0]).all().item():
            raise FloatingPointError(f"Nonfinite {mode} {name}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    # Exclude CPU copies / artifact writes / decode from generation time.
    samples = {name: output[name][0].detach().cpu() for name in ("action", "vision")}
    selection = None
    if mode == "asi":
        selection = {
            name: controller.plan[name].detach().cpu() for name in ("core_masks", "stable_mask", "execution_mask")
        }
    return samples, elapsed, summary, selection


def montage(path, task, videos):
    width, height, title = 256, 212, 24
    canvas = Image.new("RGB", (8 * width, 3 * (height + title) + 30), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"{task} | Dense trajectory chunk 3 | same input + seed", fill="black")
    for row, mode in enumerate(("dense", "asi", "toca_future")):
        for col, index in enumerate(range(4, 33, 4)):
            pixels = np.rint(videos[mode][index] * 255).astype(np.uint8)
            panel = Image.fromarray(pixels).resize((width, height), Image.Resampling.LANCZOS)
            y = 30 + row * (height + title)
            canvas.paste(panel, (col * width, y + title))
            draw.text((col * width + 4, y + 5), f"{mode} | RGB future {index}", fill="black")
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timing-repeats", type=int, default=30)
    args = parser.parse_args()
    if args.timing_repeats < 3:
        raise ValueError("Use at least three paired timing repeats")
    run, output = args.run_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    manifest = json.loads((run / "dense/manifest.json").read_text())
    requests = [json.loads(line) for line in (run / "dense/server/requests.jsonl").read_text().splitlines()]
    prompts = list(dict.fromkeys(r["prompt"] for r in requests))
    assert len(prompts) == len(manifest["tasks"]) == 10
    task_for_prompt = dict(zip(prompts, manifest["tasks"], strict=True))
    captures = sorted((run / "dense/server/captures").glob("*/sample.pt"))
    if len(captures) != 10:
        raise ValueError(f"Expected ten Dense chunk-3 captures, found {len(captures)}")
    vae = "/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
    service = EagerService(
        RobolabServerArgs(
            checkpoint_path="/root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID",
            seed=0,
            deterministic_seed=False,
            guidance=3,
            num_steps=4,
            shift=5,
            format_prompt_as_json=True,
            guardrails=False,
            output_dir=output / "model_output",
            experiment_overrides=[
                f"model.config.tokenizer.vae_path={vae}",
                "model.config.tokenizer.object_store_credential_path_pretrained=",
                "model.config.tokenizer.bucket_name=",
            ],
        )
    )
    model = service.model
    if model.tokenizer_vision_gen._keep_decoder_cache:
        raise RuntimeError("Decoder cache must reset between paired decodes")
    metrics_rows, frame_rows, latent_rows, checks = [], [], [], []
    modes = ("dense", "asi", "toca_future")
    with torch.inference_mode():
        for path in captures:
            capture = torch.load(path, map_location="cpu", weights_only=False)
            task = task_for_prompt[capture["metadata"]["prompt"]]
            assert capture["metadata"]["prompt_chunk"] == 3
            case = output / task
            case.mkdir()
            outputs, videos = {}, {}
            for mode in modes:
                values, elapsed, summary, selection = run_one(model, capture, mode)
                outputs[mode] = values
                if mode == "dense":
                    check = {"task": task}
                    for name in ("action", "vision"):
                        ref, live = capture["outputs"][name].float(), values[name].float()
                        check[name] = tensor_metrics(ref.numpy(), live.numpy())
                        if not torch.allclose(ref, live, rtol=1e-4, atol=1e-4):
                            write_json(case / "dense_equivalence_failed.json", check)
                            raise RuntimeError("Baseline checkout / current Dense replay mismatch; stop comparison")
                    checks.append(check)
                folder = case / mode
                folder.mkdir()
                torch.save(values, folder / "outputs.pt")
                if selection is not None:
                    torch.save(selection, folder / "selection.pt")
                decoded = model.decode(values["vision"].to(device="cuda")).detach().float().cpu().numpy()
                rgb = to_rgb01(decoded)
                if rgb.shape[0] != 33:
                    raise RuntimeError(f"Expected one condition + 32 future frames, got {rgb.shape}")
                videos[mode] = rgb
                write_json(
                    folder / "metadata.json",
                    {
                        "task": task,
                        "chunk": 3,
                        "mode": mode,
                        "seed": capture["kwargs"]["seed"],
                        "generation_s_unwarmed": elapsed,
                        "strategy": summary,
                        "rgb_shape": list(rgb.shape),
                        "raw_decode_min": float(decoded.min()),
                        "raw_decode_max": float(decoded.max()),
                        "rgb_mapping": "clip((decoded + 1)/2, 0, 1); same mapping for all strategies",
                    },
                )
                for frame in range(4, 33, 4):
                    Image.fromarray(np.rint(rgb[frame] * 255).astype(np.uint8)).save(folder / f"future_{frame:02d}.png")
            for mode in ("asi", "toca_future"):
                base, candidate = outputs["dense"], outputs[mode]
                latent = tensor_metrics(
                    base["vision"][:, :, 1:].float().numpy(), candidate["vision"][:, :, 1:].float().numpy()
                )
                action = tensor_metrics(base["action"][1:].float().numpy(), candidate["action"][1:].float().numpy())
                rgb = tensor_metrics(videos["dense"][1:], videos[mode][1:])
                per_frame = []
                for frame in range(1, 33):
                    fm = rgb_frame_metrics(videos["dense"][frame], videos[mode][frame])
                    per_frame.append(fm)
                    frame_rows.append({"task": task, "chunk": 3, "strategy": mode, "future_rgb_frame": frame, **fm})
                for latent_index in range(1, 9):
                    lm = tensor_metrics(
                        base["vision"][:, :, latent_index].float().numpy(),
                        candidate["vision"][:, :, latent_index].float().numpy(),
                    )
                    latent_rows.append(
                        {"task": task, "chunk": 3, "strategy": mode, "future_latent": latent_index, **lm}
                    )
                metrics_rows.append(
                    {
                        "task": task,
                        "chunk": 3,
                        "strategy": mode,
                        "rgb_cosine": rgb["cosine"],
                        "rgb_relative_l2": rgb["relative_l2"],
                        "rgb_psnr_db": float(-10 * np.log10(rgb["mse"])) if rgb["mse"] else None,
                        "rgb_ssim": statistics.mean(x["ssim"] for x in per_frame),
                        "latent_cosine": latent["cosine"],
                        "latent_relative_l2": latent["relative_l2"],
                        "action_mse": action["mse"],
                        "action_cosine": action["cosine"],
                    }
                )
            montage(case / "comparison.png", task, videos)
            write_csv(output / "paired_metrics.csv", metrics_rows)
            write_csv(output / "frame_metrics.csv", frame_rows)
            write_csv(output / "latent_frame_metrics.csv", latent_rows)
            write_json(output / "dense_equivalence.json", checks)
            print(f"[fidelity] {len(checks)}/10 {task}: saved comparison.png", flush=True)
            del decoded, videos, outputs, capture

        # One warmed, identical input, same loaded model, alternating order.
        capture = torch.load(captures[0], map_location="cpu", weights_only=False)
        for _ in range(2):
            for mode in modes:
                run_one(model, capture, mode)
        timing_rows = []
        for repeat in range(args.timing_repeats):
            order = modes[repeat % 3 :] + modes[: repeat % 3]
            for mode in order:
                _, elapsed, _, _ = run_one(model, capture, mode)
                timing_rows.append({"repeat": repeat, "strategy": mode, "generation_s": elapsed})
            if (repeat + 1) % 5 == 0:
                print(f"[timing] {repeat + 1}/{args.timing_repeats} alternating rounds", flush=True)
        write_csv(output / "timing_samples.csv", timing_rows)
    timing = {}
    for mode in modes:
        durations = [r["generation_s"] for r in timing_rows if r["strategy"] == mode]
        timing[mode] = {
            "mean_s": statistics.mean(durations),
            "median_s": statistics.median(durations),
            "p90_s": float(np.quantile(durations, 0.9)),
            "count": len(durations),
        }
    for value in timing.values():
        value["median_speedup_vs_dense"] = timing["dense"]["median_s"] / value["median_s"]
    aggregate = {}
    for mode in ("asi", "toca_future"):
        rows = [r for r in metrics_rows if r["strategy"] == mode]
        aggregate[mode] = {
            name: statistics.mean(r[name] for r in rows)
            for name in rows[0]
            if name not in ("task", "chunk", "strategy")
        }
        aggregate[mode]["samples"] = len(rows)
    write_json(
        output / "summary.json",
        {
            "status": "complete",
            "samples": 10,
            "frame_comparisons": len(frame_rows),
            "latent_comparisons": len(latent_rows),
            "aggregation": "Per-chunk metrics first, macro mean across ten Dense chunk-3 samples; RGB SSIM is per-frame mean",
            "reference": "Same-input Dense prediction, NOT real future video ground truth",
            "fidelity": aggregate,
            "timing": timing,
            "timing_input": str(captures[0]),
            "timing_includes": "generation + controller + required validation; excludes CPU copies, decode, simulator and artifact IO",
            "all_finite": True,
            "dense_cross_checkout_equivalence_passed": len(checks),
        },
    )
    print(json.dumps({"fidelity": aggregate, "timing": timing}, indent=2), flush=True)


if __name__ == "__main__":
    main()
