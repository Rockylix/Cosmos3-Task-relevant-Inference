"""B0 vs B1 paired single-chunk benchmark; optional separate NVTX trace run."""

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

from cosmos_framework.inference import edge_core_stable_fast as fast
from cosmos_framework.inference.asi_batch_only import BatchOnlyController, batch_only_profiles
from cosmos_framework.scripts import robolab_version1 as legacy
from cosmos_framework.scripts.action_policy_server_robolab_version1 import Version1PolicyService, Version1ServerArgs

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = Path(
    "/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/experiments/asi_smoke10_s0_p0_v1/_temporary_inputs/BananaInBowlTask.pt"
)
VAE = "/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"


def metrics(value, reference):
    a, b = value.double().flatten(), reference.double().flatten()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Nonfinite or shape mismatch")
    return dict(
        mse=float((a - b).square().mean()),
        max_abs=float((a - b).abs().max()),
        relative_l2=float((a - b).norm() / b.norm().clamp_min(1e-24)),
        cosine=float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-24)),
        exact=torch.equal(a, b),
    )


class Instrumentation:
    """Identical optional instrumentation for both controllers; timing runs disable trace."""

    trace = False
    audit = False

    def scope(self, name):
        return torch.cuda.nvtx.range(name) if self.trace else nullcontext()

    def _capture_profile(self, **kwargs):
        with self.scope("asi.score"):
            super()._capture_profile(**kwargs)
        if self.audit:
            alternate = batch_only_profiles(
                torch=torch,
                **{k: kwargs[k] for k in ("q_gen", "k_ar", "k_gen", "v_ar", "v_gen")},
                scaling=float(kwargs["scaling"]),
                token_layout=self._layout,
            )
            self.score_audit.append(
                dict(block=kwargs["layer_index"], **metrics(alternate, self._profile_records[-1]["profiles"]))
            )

    def run_layer(self, **kwargs):
        label = f"asi.step{self._current['step']}.{self._current['branch']}.B{kwargs['block']:02d}"
        with self.scope(label):
            return super().run_layer(**kwargs)

    def _slice_pack_and_rope(self, *args, **kwargs):
        with self.scope("asi.pack"):
            return super()._slice_pack_and_rope(*args, **kwargs)

    def end_stack(self, *args, **kwargs):
        first = self._current == dict(step=0, branch="conditional")
        with self.scope("asi.select" if first else "asi.restore"):
            return super().end_stack(*args, **kwargs)


class B0(Instrumentation, legacy.Version1Controller):
    pass


class B1(Instrumentation, BatchOnlyController):
    pass


def controllers():
    from cosmos_framework.inference.asi_existing_ablation import CONTROLLERS
    from cosmos_framework.inference.asi_main_lse import MainLSEController

    return {
        "b0": B0,
        "b1": B1,
        "native": legacy.Version1Controller,
        **{
            key: type(key.upper(), (Instrumentation, cls), {})
            for key, cls in {**CONTROLLERS, "b6": MainLSEController}.items()
        },
    }


def require_equal_selection_output(diff):
    if not (
        diff["core_blocks_exact"]
        and all(diff[k + "_xor"] == 0 for k in ("core_masks", "stable_mask", "execution_mask"))
        and diff["action"]["exact"]
        and diff["future_vision"]["exact"]
    ):
        raise RuntimeError("Changed discrete selection or output: stop for user review; gate.json saved")


def compare(a, b):
    result = {
        k: metrics(a[k], b[k])
        for k in ("action", "future_vision", "raw_profiles", "block_mass", "block_entropy", "block_quality")
    }
    for key in ("core_masks", "stable_mask", "execution_mask"):
        result[key + "_xor"] = int((a[key] != b[key]).sum())
    result["core_blocks_exact"] = a["core_blocks"] == b["core_blocks"]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--capture", type=Path, default=CAPTURE)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--modes", nargs="+", default=["b0", "b1"])
    p.add_argument("--trace-mode")
    args = p.parse_args()
    if args.warmup < 5 or args.repeats < 1:
        p.error("At least five warmups and one repeat")
    args.output.mkdir(parents=True, exist_ok=False)
    capture = torch.load(args.capture, map_location="cpu", weights_only=False)
    assert all(capture["kwargs"][k] == v for k, v in dict(num_steps=4, shift=5, guidance=3).items())
    service = Version1PolicyService(
        Version1ServerArgs(
            checkpoint_path="/root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID",
            asi_execution="legacy",
            seed=0,
            deterministic_seed=False,
            guidance=3,
            num_steps=4,
            shift=5,
            guardrails=False,
            decode_video=False,
            format_prompt_as_json=True,
            output_dir=args.output / "model_output",
            experiment_overrides=[
                f"model.config.tokenizer.vae_path={VAE}",
                "model.config.tokenizer.object_store_credential_path_pretrained=",
                "model.config.tokenizer.bucket_name=",
            ],
        )
    )
    model = service.model
    generate = type(model).generate_samples_from_batch.__get__(model)
    assert not model.net.pad_for_cuda_graphs
    assert not any(hasattr(layer, "_orig_mod") for layer in model.net.language_model.model.layers)

    def run(mode, offset=0, audit=False, trace=False):
        positional, kwargs = copy.deepcopy(capture["args"]), copy.deepcopy(capture["kwargs"])
        kwargs["seed"] = [int(kwargs["seed"][0]) + offset]
        seed = kwargs["seed"][0]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cls = controllers()[mode]
        ctrl = cls(torch=torch, net=model.net, guidance=3, num_steps=4)
        ctrl.trace, ctrl.audit, ctrl.score_audit = trace, audit, []
        original_attention = legacy.attention
        original_fast_attention, original_fast_score = fast.attention, fast.action_aligned_future_profiles
        if trace:
            # NVTX-only annotation around the unchanged scorer attention API, both modes.
            def annotate_attention(*a, **kw):
                with torch.cuda.nvtx.range("asi.score.lse_attention"):
                    return original_attention(*a, **kw)

            legacy.attention = annotate_attention
            fast.attention = annotate_attention

            def annotate_fast_score(*a, **kw):
                with torch.cuda.nvtx.range("asi.score.fast"):
                    return original_fast_score(*a, **kw)

            fast.action_aligned_future_profiles = annotate_fast_score
        try:
            with torch.inference_mode(), ctrl:
                torch.cuda.synchronize()
                if trace:
                    torch.cuda.cudart().cudaProfilerStart()
                    torch.cuda.nvtx.range_push("asi.chunk.generate")
                started = time.perf_counter()
                samples = generate(*positional, **kwargs)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if trace:
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.cudart().cudaProfilerStop()
        finally:
            legacy.attention = original_attention
            fast.attention, fast.action_aligned_future_profiles = original_fast_attention, original_fast_score
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
        for value in result.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise FloatingPointError(mode)
        return result, elapsed, summary, ctrl.score_audit

    files = [
        "cosmos_framework/inference/asi_batch_only.py",
        "tools/benchmark_asi_batch_only.py",
        "cosmos_framework/scripts/robolab_version1.py",
        "cosmos_framework/model/generator/mot/unified_mot.py",
        "cosmos_framework/inference/asi_existing_ablation.py",
        "cosmos_framework/inference/edge_core_stable.py",
        "cosmos_framework/inference/edge_core_stable_fast.py",
        "cosmos_framework/inference/asi_main_lse.py",
        "cosmos_framework/model/generator/mot/attention.py",
        "cosmos_framework/model/generator/mot/inference_text_kv_memory.py",
    ]
    report = dict(
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files},
        capture=str(args.capture),
        capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
        python=sys.executable,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        compile=False,
        cuda_graphs=False,
        seed=capture["kwargs"]["seed"],
        steps=4,
        shift=5,
        guidance=3,
        timing_scope="generate_samples_from_batch + final CUDA synchronize; fresh controller/input/RNG each run; no decode/RPC/CPU summaries",
        capture_metadata=capture.get("metadata"),
        trace_mode=args.trace_mode,
    )
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2, default=str))
    modes = [args.trace_mode] if args.trace_mode else args.modes
    if not args.trace_mode and "b0" not in modes:
        p.error("B0 must be included as the fixed legacy reference")
    latest = {}
    for mode in modes:
        for i in range(args.warmup):
            latest[mode] = run(mode)
            print("WARMUP", mode, i + 1, latest[mode][1], flush=True)
    if args.trace_mode:
        result, elapsed, summary, _ = run(args.trace_mode, trace=True)
        diff = compare(result, latest[args.trace_mode][0])
        require_equal_selection_output(diff)
        report.update(trace_wall_s=elapsed, trace_vs_warm=diff, summary=summary)
    else:
        baseline, _, _, score_audit = run("b0", audit=True)
        native = run("native")[0]
        native_check = compare(baseline, native)
        assert (
            native_check["action"]["exact"]
            and native_check["future_vision"]["exact"]
            and native_check["execution_mask_xor"] == 0
        )
        report.update(
            native_baseline_check=native_check, same_qk_scoring=score_audit, comparisons={}, timing={}, summaries={}
        )
        if "b6" in modes:
            # Separate untimed correctness audit on the actual same-Q/K tensors.
            audit_records = []
            original_score = fast.action_aligned_future_profiles

            def audit_main_lse(q, ka, kg, va, vg, scale, geometry, *, action_lse=None):
                if action_lse is None:
                    raise RuntimeError("B6 silently fell back to extra scorer attention")
                start = geometry[0]
                out_main, lse_main = fast.attention(
                    query=q.unsqueeze(0),
                    key=torch.cat((ka, kg)).unsqueeze(0),
                    value=torch.cat((va, vg)).unsqueeze(0),
                    scale=scale,
                    return_lse=True,
                )
                out_plain = fast.attention(
                    query=q.unsqueeze(0),
                    key=torch.cat((ka, kg)).unsqueeze(0),
                    value=torch.cat((va, vg)).unsqueeze(0),
                    scale=scale,
                )
                _, lse_old = fast.attention(
                    query=q[start : start + 32].unsqueeze(0),
                    key=torch.cat((ka, kg)).unsqueeze(0),
                    value=torch.cat((va, vg)).unsqueeze(0),
                    scale=scale,
                    return_lse=True,
                )
                result = original_score(q, ka, kg, va, vg, scale, geometry, action_lse=action_lse)
                reference = original_score(q, ka, kg, va, vg, scale, geometry, action_lse=lse_old)
                record = dict(
                    block=len(audit_records),
                    main_output_on_off=metrics(out_main, out_plain),
                    reused_lse_vs_recomputed_main=metrics(action_lse, lse_main[:, start : start + 32]),
                    main_lse_vs_32q=metrics(action_lse, lse_old),
                    profiles=metrics(result, reference),
                )
                audit_records.append(record)
                assert record["main_output_on_off"]["exact"]
                assert record["reused_lse_vs_recomputed_main"]["exact"]
                return result

            fast.action_aligned_future_profiles = audit_main_lse
            try:
                run("b6")
            finally:
                fast.action_aligned_future_profiles = original_score
            report["main_lse_audit"] = audit_records
            assert len(audit_records) == 28
        for offset in (0, 1):
            paired = {m: run(m, offset=offset)[0] for m in modes}
            diffs = {m: compare(paired[m], paired["b0"]) for m in modes if m != "b0"}
            report["comparisons"][str(offset)] = diffs
            torch.save(paired, args.output / f"outputs_seed_offset{offset}.pt")
            (args.output / "gate.json").write_text(json.dumps(report, indent=2, default=str, allow_nan=False))
            for diff in diffs.values():
                require_equal_selection_output(diff)
        times = {m: [] for m in modes}
        for rep in range(args.repeats):
            for mode in modes if rep % 2 == 0 else list(reversed(modes)):
                result, elapsed, summary, _ = run(mode)
                diff = compare(result, latest[mode][0])
                require_equal_selection_output(diff)
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
        report["speedup_median"] = {
            m: report["timing"]["b0"]["median_s"] / report["timing"][m]["median_s"] for m in modes
        }
    (args.output / "results.json").write_text(json.dumps(report, indent=2, default=str, allow_nan=False))
    print("COMPLETE", args.output, flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
