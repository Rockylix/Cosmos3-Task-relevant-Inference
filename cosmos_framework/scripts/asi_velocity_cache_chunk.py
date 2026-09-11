"""One paired DROID chunk: Dense / ASI / dense-step0 ASI / ASI velocity cache."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.inference.asi_velocity_cache import CacheSamplerAdapter

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path("/root/robolab")
CURRENT = PROJECT / "cosmos-framework-edge-core80-stable104-action-weighted"
CAPTURE = CURRENT / "experiments/asi_smoke10_s0_p0_v1/_temporary_inputs/BananaInBowlTask.pt"
VAE = Path(
    "/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/"
    "921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
)
MODES = ("dense", "asi", "asi_dense_step0", "asi_velocity_cache")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def metrics(value, reference):
    if value.shape != reference.shape or value.numel() == 0:
        raise ValueError("Bad metric shape")
    value, reference = value.double().reshape(-1), reference.double().reshape(-1)
    if not torch.isfinite(value).all() or not torch.isfinite(reference).all():
        raise FloatingPointError("Nonfinite metric input")
    difference = value - reference
    cosine = float(value.dot(reference) / (value.norm() * reference.norm()).clamp_min(1e-24))
    if not -1 - 1e-12 <= cosine <= 1 + 1e-12:
        raise ArithmeticError("Invalid cosine")
    return {
        "mse": float(difference.square().mean()),
        "cosine": max(-1.0, min(1.0, cosine)),
        "relative_l2": float(difference.norm() / reference.norm().clamp_min(1e-12)),
    }


def run_mode(model, capture, mode, *, core_block_count=6, fixed_core_blocks=None):
    from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq
    from cosmos_framework.scripts.robolab_version1 import Version1Controller

    positional, kwargs = copy.deepcopy(capture["args"]), copy.deepcopy(capture["kwargs"])
    seed = kwargs["seed"][0]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    controller = (
        None
        if mode == "dense"
        else Version1Controller(
            torch=torch,
            net=model.net,
            guidance=3,
            num_steps=4,
            dense_step0=mode in ("asi_dense_step0", "asi_velocity_cache"),
            core_block_count=core_block_count,
            fixed_core_blocks=fixed_core_blocks,
        )
    )
    layout, rows, intermediate_finite, handles = {}, {}, [], []

    def layout_hook(module, args, keywords):
        if keywords.get("und_only", args[2] if len(args) > 2 else False):
            return
        packed = args[0] if args else keywords["packed_seq"]
        observed = {
            "vision_shape": list(packed.vision.tokens[0].shape),
            "grid_thw": list(packed.vision.token_shapes[0]),
        }
        if layout and layout != observed:
            raise RuntimeError("Native layout changed within the request")
        layout.update(observed)

    handles.append(model.net.register_forward_pre_hook(layout_hook, with_kwargs=True))
    for block, layer in enumerate(model.net.language_model.model.layers):
        for name in ("q", "k", "v", "o", "mlp"):
            module = layer.mlp_moe_gen if name == "mlp" else getattr(layer.self_attn, f"{name}_proj_moe_gen")
            key = f"B{block}/{name}"
            rows[key] = []
            handles.append(module.register_forward_pre_hook(lambda m, a, key=key: rows[key].append(len(a[0]))))
        handles.append(
            layer.register_forward_hook(
                lambda m, a, out: intermediate_finite.append(torch.isfinite(get_gen_seq(out[0])).all())
            )
        )
    sampler = CacheSamplerAdapter(
        model.sampler,
        controller,
        layout,
        int(model.config.diffusion_expert_config.patch_spatial),
        enabled=mode in ("asi_velocity_cache", "asi_cached_step0"),
        step0_source="asi_cfg" if mode == "asi_cached_step0" else "dense_cfg",
    )
    try:
        with torch.inference_mode(), controller if controller is not None else nullcontext():
            result = model.generate_samples_from_batch(*positional, sampler=sampler, **kwargs)
        summary = controller.finish() if controller is not None else {}
    finally:
        for handle in reversed(handles):
            handle.remove()
    expected = (
        [3093] * 8
        if mode == "dense"
        else ([3093] + [1845] * 7 if mode in ("asi", "asi_cached_step0") else [3093] * 2 + [1845] * 6)
    )
    if len(rows) != 140 or any(value != expected for value in rows.values()):
        raise AssertionError(f"Actual Q/K/V/O/MLP token rows incorrect: {rows}")
    if len(intermediate_finite) != 224 or not torch.stack(intermediate_finite).all():
        raise FloatingPointError("Nonfinite block output or incorrect block count")
    outputs = {key: result[key][0].detach().cpu().clone() for key in ("action", "vision")}
    if not all(torch.isfinite(value).all() for value in outputs.values()):
        raise FloatingPointError("Nonfinite final model output")
    selection = (
        None
        if controller is None
        else {
            key: controller.plan[key].detach().cpu()
            for key in ("execution_mask", "core_masks", "stable_mask", "block_quality")
        }
    )
    return (
        outputs,
        sampler.initial_velocity,
        selection,
        {
            "controller": summary,
            "layout": layout,
            "steps": sampler.records,
            "expected_gen_rows_per_module": expected,
            "all_140_module_row_checks_passed": True,
            "all_224_block_outputs_finite": True,
        },
    )


def save_frames(folder, rgb):
    from PIL import Image, ImageDraw

    folder.mkdir()
    pixels = (rgb[0].permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)
    thumbs = []
    for index, frame in enumerate(pixels, 1):
        picture = Image.fromarray(frame)
        picture.save(folder / f"frame_{index:03d}.png")
        thumb = picture.copy()
        thumb.thumbnail((240, 210))
        thumbs.append(thumb)
    sheet = Image.new("RGB", (8 * 240, 4 * 234), "white")
    draw = ImageDraw.Draw(sheet)
    for index, picture in enumerate(thumbs):
        x, y = index % 8 * 240, index // 8 * 234
        draw.text((x + 4, y + 3), f"future RGB {index + 1}", fill="black")
        sheet.paste(picture, (x, y + 22))
    sheet.save(folder.parent / "all_future_frames.png")


def save_comparison(output):
    from PIL import Image, ImageDraw

    examples = (1, 4, 8, 12, 16, 20, 24, 32)
    sheet = Image.new("RGB", (200 + 8 * 200, 4 * 190), "white")
    draw = ImageDraw.Draw(sheet)
    for row, mode in enumerate(MODES):
        draw.text((5, row * 190 + 70), mode, fill="black")
        for col, frame in enumerate(examples):
            with Image.open(output / mode / "future_frames" / f"frame_{frame:03d}.png") as original:
                thumbnail = original.copy()
            thumbnail.thumbnail((200, 165))
            x, y = 200 + col * 200, row * 190
            draw.text((x + 3, y + 3), f"RGB {frame}", fill="black")
            sheet.paste(thumbnail, (x, y + 23))
    sheet.save(output / "comparison.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=CAPTURE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Check before model imports/initialization or creating any result-looking artifacts.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: repair host driver/library mismatch before inference")
    if args.output.exists():
        raise FileExistsError("Use a new output directory; never overwrite an earlier experiment")
    for path in (args.capture, VAE, PROJECT / "RoboLab/Cosmos3-Edge-Policy-DROID"):
        if not path.exists():
            raise FileNotFoundError(path)
    from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService, RobolabServerArgs

    class EagerService(RobolabPolicyService):
        def _build_setup_args(self, setup_args):
            return (
                super()
                ._build_setup_args(setup_args)
                .model_copy(update={"use_torch_compile": False, "use_cuda_graphs": False})
            )

    capture = torch.load(args.capture, weights_only=False, map_location="cpu")
    if capture["kwargs"] != {"guidance": 3.0, "seed": [1097657232], "num_steps": 4, "shift": 5.0}:
        raise ValueError("This first experiment is frozen to Banana c3 / shift5 / original request seed")
    output = args.output.resolve()
    output.mkdir(parents=True)
    source_files = [
        Path(__file__),
        ROOT / "cosmos_framework/scripts/robolab_version1.py",
        ROOT / "cosmos_framework/inference/asi_velocity_cache.py",
    ]
    write_json(
        output / "manifest.json",
        {
            "phase": "started",
            "capture": str(args.capture.resolve()),
            "capture_sha256": hashlib.sha256(args.capture.read_bytes()).hexdigest(),
            "capture_task_chunk": {k: capture["metadata"][k] for k in ("task", "chunk", "seed")},
            "kwargs": capture["kwargs"],
            "modes": MODES,
            "compile": False,
            "cuda_graphs": False,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "pythonpath": os.environ.get("PYTHONPATH"),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files
            },
        },
    )
    service = EagerService(
        RobolabServerArgs(
            checkpoint_path=str(PROJECT / "RoboLab/Cosmos3-Edge-Policy-DROID"),
            guardrails=False,
            output_dir=output / "model_output",
            format_prompt_as_json=True,
            experiment_overrides=[
                f"model.config.tokenizer.vae_path={VAE}",
                "model.config.tokenizer.object_store_credential_path_pretrained=",
                "model.config.tokenizer.bucket_name=",
            ],
        )
    )
    if service.model.config.rectified_flow_inference_config.scheduler_type != "unipc":
        raise RuntimeError("Expected the native UniPC scheduler")
    reference_outputs = reference_rgb = reference_initial = common_mask = None
    metrics_by_mode = {}
    for mode in MODES:
        print(f"[velocity-cache] {mode}", flush=True)
        predictions, initial, selection, audit = run_mode(service.model, capture, mode)
        if mode == "asi":
            for key in predictions:
                torch.testing.assert_close(predictions[key], capture["outputs"][key], rtol=1e-4, atol=1e-4)
        if mode in ("asi_dense_step0", "asi_velocity_cache"):
            torch.testing.assert_close(initial, reference_initial, rtol=1e-5, atol=1e-5)
        if selection is not None:
            if common_mask is None:
                common_mask = selection["execution_mask"]
            elif not torch.equal(common_mask, selection["execution_mask"]):
                raise AssertionError("A/B/C did not select the identical Step0 conditional mask")
        folder = output / mode
        folder.mkdir()
        with torch.inference_mode():
            decoded = service.model.decode(predictions["vision"].cuda()).float().cpu()
        if decoded.ndim != 5 or tuple(decoded.shape[:3]) != (1, 3, 33) or not torch.isfinite(decoded).all():
            raise ValueError("Expected 33 finite decoded frames including the condition frame")
        rgb = (decoded[:, :, 1:].clamp(-1, 1) + 1) / 2
        if mode == "dense":
            reference_outputs, reference_rgb, reference_initial = predictions, rgb, initial
        metrics_by_mode[mode] = {
            "action": metrics(predictions["action"][1:], reference_outputs["action"][1:]),
            "future_latent": metrics(predictions["vision"][:, :, 1:], reference_outputs["vision"][:, :, 1:]),
            "future_rgb": metrics(rgb, reference_rgb),
        }
        torch.save(
            {"outputs": predictions, "selection": selection, "step0_guided_velocity": initial}, folder / "outputs.pt"
        )
        write_json(folder / "audit.json", audit)
        save_frames(folder / "future_frames", rgb)
        write_json(output / "metrics.json", metrics_by_mode)
    save_comparison(output)
    write_json(
        output / "completion.json",
        {
            "complete": True,
            "paired_modes": list(MODES),
            "same_masks": True,
            "all_module_row_checks_passed": True,
            "all_outputs_finite": True,
            "note": "One offline chunk only; no closed-loop success or performance claim",
        },
    )
    print(f"[velocity-cache] COMPLETE: {output / 'comparison.png'}", flush=True)


if __name__ == "__main__":
    main()
