"""Same-input reference/joint ToCa correctness, future fidelity, and warm timing."""

from cosmos_framework.inference.common.init import init_script

init_script()

import argparse
import copy
import json
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq, get_und_seq
from cosmos_framework.inference.future_fidelity_metrics import rgb_frame_metrics, tensor_metrics, to_rgb01
from cosmos_framework.inference.toca_future import (
    ToCaFutureConfig,
    ToCaFutureController,
    incoming_future_score,
    select_fresh,
)
from cosmos_framework.model.attention import attention
from cosmos_framework.scripts.action_policy_server_robolab import RobolabServerArgs
from cosmos_framework.scripts.paired_future_fidelity import EagerService, write_csv, write_json


class AuditController(ToCaFutureController):
    def __init__(self, net):
        super().__init__(net, ToCaFutureConfig(attention_backend="joint"))
        self.reference_scores = {}
        self.checks = []
        self.selection_checks = []

    def _joint_dispatch(self, block, q_pack, k_pack, v_pack, mask, **kwargs):
        output = super()._joint_dispatch(block, q_pack, k_pack, v_pack, mask, **kwargs)
        q, kg, vg = get_gen_seq(q_pack), get_gen_seq(k_pack), get_gen_seq(v_pack)
        memory = kwargs.get("memory_value")
        if memory is not None and memory.und_k_cached is not None:
            ku, vu = memory.und_k_cached[0], memory.und_v_cached[0]
        else:
            kp = kwargs.get("packed_key_states_normalized")
            ku, vu = get_und_seq(kp if kp is not None else k_pack), get_und_seq(v_pack)
        step, branch = self.current
        ref = incoming_future_score(q, ku, kg, self.future, self.layers[block].self_attn.scaling)
        self.reference_scores[branch, block] = ref
        live = self.scores[branch, block]
        native = attention(
            query=q[None], key=torch.cat((ku, kg))[None], value=torch.cat((vu, vg))[None], is_causal=False
        )[0].flatten(-2, -1)
        gen_out = get_gen_seq(output[0])
        score_rel = float((live - ref).norm() / ref.norm())
        score_max = float((live - ref).abs().max())
        av_rel = float((gen_out.float() - native.float()).norm() / native.float().norm())
        torch.testing.assert_close(live, ref, rtol=2e-4, atol=2e-7)
        assert av_rel < 0.005, (step, branch, block, av_rel)
        self.checks.append(
            {
                "step": step,
                "branch": branch,
                "block": block,
                "score_relative_l2": score_rel,
                "score_max_absolute": score_max,
                "av_relative_l2": av_rel,
            }
        )
        return output

    def _cached(self, block, pack, rope, memory, gen_only):
        step, branch = self.current
        if branch == "conditional":
            scores = (self.reference_scores["conditional", block] + self.reference_scores["unconditional", block]) * 0.5
            expected, _ = select_fresh(
                scores, self.ages[block], self.config.fresh_count(block, 28, len(self.future)), self.config
            )
        output = super()._cached(block, pack, rope, memory, gen_only)
        if branch == "conditional":
            actual = self.indices[step, block]
            overlap = float(torch.isin(actual, expected).float().mean())
            self.selection_checks.append(
                {
                    "step": step,
                    "block": block,
                    "count": len(actual),
                    "set_overlap": overlap,
                    "identical_set": bool(torch.equal(actual.sort().values, expected.sort().values)),
                }
            )
        return output


def generate(model, capture, mode, audit=False):
    args, kwargs = copy.deepcopy(capture["args"]), copy.deepcopy(capture["kwargs"])
    torch.cuda.synchronize()
    start = time.perf_counter()
    controller = None
    if mode != "dense":
        controller = (
            AuditController(model.net)
            if audit
            else ToCaFutureController(model.net, ToCaFutureConfig(attention_backend=mode))
        )
    with controller if controller is not None else nullcontext():
        output = model.generate_samples_from_batch(*args, **kwargs)
    summary = controller.finish() if controller is not None else {}
    assert all(torch.isfinite(output[name][0]).all() for name in ("action", "vision"))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = {name: output[name][0].detach().cpu() for name in ("action", "vision")}
    return result, elapsed, summary, controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--decode", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = args.source_run.resolve()
    torch.set_num_threads(4)
    manifest = json.loads((source / "dense/manifest.json").read_text())
    req = [json.loads(x) for x in (source / "dense/server/requests.jsonl").read_text().splitlines()]
    task_map = dict(zip(dict.fromkeys(x["prompt"] for x in req), manifest["tasks"], strict=True))
    captures = sorted((source / "dense/server/captures").glob("*/sample.pt"))[: args.tasks]
    vae = "/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
    model = EagerService(
        RobolabServerArgs(
            checkpoint_path="/root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID",
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
    checks = []
    fidelity = []
    frames = []
    block_checks = []
    selection_checks = []
    with torch.inference_mode():
        for index, path in enumerate(captures):
            cap = torch.load(path, map_location="cpu", weights_only=False)
            task = task_map[cap["metadata"]["prompt"]]
            folder = out / task
            folder.mkdir()
            joint, _, summary, control = generate(model, cap, "joint", audit=True)
            for row in control.checks:
                block_checks.append({"task": task, **row})
            for row in control.selection_checks:
                selection_checks.append({"task": task, **row})
            del control
            ref, _, _, _ = generate(model, cap, "reference")
            dense = cap["outputs"]
            old = torch.load(
                source / "fidelity" / task / "toca_future/outputs.pt", map_location="cpu", weights_only=False
            )
            for name in ("action", "vision"):
                assert torch.equal(ref[name], old[name]), ("Reference path changed", task, name)
            torch.save(joint, folder / "joint_outputs.pt")
            checks.append({"task": task, "old_reference_exact": True, "strategy": summary})
            for name in ("action", "vision"):
                j, r, d = joint[name].float().numpy(), ref[name].float().numpy(), dense[name].float().numpy()
                j, r, d = (x[1:] if name == "action" else x[:, :, 1:] for x in (j, r, d))
                for reference, value in (("old_toca", r), ("dense", d)):
                    fidelity.append(
                        {"task": task, "modality": name, "reference": reference, **tensor_metrics(value, j)}
                    )
            if args.decode:
                videos = {}
                for mode, values in (("dense", dense), ("old_toca", ref), ("joint", joint)):
                    videos[mode] = to_rgb01(model.decode(values["vision"].cuda()).float().cpu().numpy())
                    dest = folder / mode
                    dest.mkdir()
                    for f in range(4, 33, 4):
                        Image.fromarray(np.rint(videos[mode][f] * 255).astype(np.uint8)).save(
                            dest / f"future_{f:02d}.png"
                        )
                for reference in ("dense", "old_toca"):
                    fm = [rgb_frame_metrics(videos[reference][f], videos["joint"][f]) for f in range(1, 33)]
                    rgb = tensor_metrics(videos[reference][1:], videos["joint"][1:])
                    fidelity.append({"task": task, "modality": "rgb", "reference": reference, **rgb})
                    frames.extend({"task": task, "reference": reference, "frame": f, **m} for f, m in enumerate(fm, 1))
                canvas = Image.new("RGB", (8 * 256, 3 * 236), "white")
                draw = ImageDraw.Draw(canvas)
                for row, mode in enumerate(("dense", "old_toca", "joint")):
                    for col, f in enumerate(range(4, 33, 4)):
                        panel = Image.fromarray(np.rint(videos[mode][f] * 255).astype(np.uint8)).resize((256, 212))
                        canvas.paste(panel, (col * 256, row * 236 + 24))
                        draw.text((col * 256 + 4, row * 236 + 5), f"{mode} future {f}", fill="black")
                canvas.save(folder / "comparison.png")
                del videos
            write_json(out / "checks.json", checks)
            write_csv(out / "block_correctness.csv", block_checks)
            write_csv(out / "selection_correctness.csv", selection_checks)
            write_csv(out / "fidelity.csv", fidelity)
            write_csv(out / "frame_metrics.csv", frames)
            print(f"[validation] {index + 1}/{len(captures)} {task}", flush=True)
        cap = torch.load(captures[0], map_location="cpu", weights_only=False)
        modes = ("dense", "reference", "joint")
        for _ in range(2):
            for mode in modes:
                generate(model, cap, mode)
        timing = []
        for repeat in range(args.repeats):
            order = modes[repeat % 3 :] + modes[: repeat % 3]
            for mode in order:
                _, seconds, _, _ = generate(model, cap, mode)
                timing.append({"repeat": repeat, "mode": mode, "generation_s": seconds})
            if repeat % 5 == 4:
                print(f"[timing] {repeat + 1}/{args.repeats}", flush=True)
        write_csv(out / "timing.csv", timing)
    stats = {}
    for mode in modes:
        x = [r["generation_s"] for r in timing if r["mode"] == mode]
        stats[mode] = {
            "mean_s": statistics.mean(x),
            "median_s": statistics.median(x),
            "p90_s": float(np.quantile(x, 0.9)),
        }
    for row in stats.values():
        row["speedup"] = stats["dense"]["median_s"] / row["median_s"]
    summary = {
        "status": "complete",
        "tasks": len(captures),
        "timing": stats,
        "score_relative_l2_max": max(r["score_relative_l2"] for r in block_checks),
        "score_max_absolute": max(r["score_max_absolute"] for r in block_checks),
        "av_relative_l2_max": max(r["av_relative_l2"] for r in block_checks),
        "topk_identical_sets": sum(r["identical_set"] for r in selection_checks),
        "topk_total": len(selection_checks),
        "topk_min_overlap": min(r["set_overlap"] for r in selection_checks),
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
