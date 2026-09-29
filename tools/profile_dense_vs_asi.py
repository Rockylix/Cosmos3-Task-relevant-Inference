"""Separate-checkout, eager-only warmed Dense/ASI trace; no model edits."""

from cosmos_framework.inference.common.init import init_script

init_script()

import argparse
import copy
import hashlib
import json
import random
import subprocess
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from cosmos_framework.scripts import action_policy_server_robolab as server

CAPTURE = Path(
    "/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/experiments/asi_smoke10_s0_p0_v1/_temporary_inputs/BananaInBowlTask.pt"
)
VAE = "/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"


class EagerService(server.RobolabPolicyService):
    def _build_setup_args(self, args):
        return super()._build_setup_args(args).model_copy(update={"use_torch_compile": False, "use_cuda_graphs": False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("dense", "asi"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-eager", type=Path, help="Optional historical compile-benchmark outputs file (key: eager)")
    args = p.parse_args()
    old_eager = None
    if args.reference_eager is not None:
        if args.mode != "asi":
            p.error("--reference-eager applies only to ASI")
        old_eager = torch.load(args.reference_eager, map_location="cpu", weights_only=False)["eager"]
    expected = Path(
        "/root/robolab/cosmos-framework-edge" if args.mode == "dense" else "/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted"
    )
    source = Path(server.__file__).resolve()
    assert source.is_relative_to(expected), (source, expected)
    args.output.mkdir(parents=True, exist_ok=False)
    capture = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    assert all(capture["kwargs"][k] == v for k, v in dict(num_steps=4, guidance=3, shift=5).items())
    service = EagerService(
        server.RobolabServerArgs(
            checkpoint_path="/root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID",
            num_steps=4,
            guidance=3,
            shift=5,
            guardrails=False,
            format_prompt_as_json=True,
            decode_video=False,
            output_dir=args.output / "model_output",
            experiment_overrides=[
                f"model.config.tokenizer.vae_path={VAE}",
                "model.config.tokenizer.object_store_credential_path_pretrained=",
                "model.config.tokenizer.bucket_name=",
            ],
        )
    )
    model, net = service.model, service.model.net
    assert not model.config.compile.enabled
    assert not net.pad_for_cuda_graphs
    layers = list(net.language_model.model.layers)
    assert len(layers) == 28 and not any(hasattr(x, "_orig_mod") for x in layers)
    records, score_calls = [], []

    @contextmanager
    def instrument(ctrl):
        with ExitStack() as stack:

            def layer_wrapper(original, block):
                def call(*a, **kw):
                    i = len(records)
                    assert i % 28 == block
                    step, branch = i // 56, ("conditional" if (i // 28) % 2 == 0 else "unconditional")
                    pack = kw.get("input", a[0] if a else None)
                    record = dict(step=step, branch=branch, block=block)
                    if isinstance(pack, dict):
                        for key in ("_num_full_tokens", "_num_causal_tokens"):
                            record[key] = pack.get(key)
                    records.append(record)
                    with torch.cuda.nvtx.range(f"asi.step{step}.{branch}.B{block:02d}"):
                        return original(*a, **kw)

                return call

            for block, layer in enumerate(layers):
                stack.enter_context(patch.object(layer, "forward", layer_wrapper(layer.forward, block)))
            if ctrl is not None:
                from cosmos_framework.inference import edge_core_stable_fast as fast

                original_score = fast.action_aligned_future_profiles

                def score(*a, **kw):
                    score_calls.append(len(records) - 1)
                    with torch.cuda.nvtx.range("asi.score"):
                        return original_score(*a, **kw)

                stack.enter_context(patch.object(fast, "action_aligned_future_profiles", score))
                original_end = ctrl.end_stack

                def end(*a, **kw):
                    label = "asi.select" if ctrl._current == dict(step=0, branch="conditional") else "asi.restore"
                    with torch.cuda.nvtx.range(label):
                        return original_end(*a, **kw)

                stack.enter_context(patch.object(ctrl, "end_stack", end))
            yield

    def run(trace=False):
        positional, kwargs = copy.deepcopy(capture["args"]), copy.deepcopy(capture["kwargs"])
        seed = int(kwargs["seed"][0])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ctrl = None
        if args.mode == "asi":
            from cosmos_framework.inference.asi_main_lse import MainLSEController

            ctrl = MainLSEController(torch=torch, net=net, guidance=3, num_steps=4)
        with torch.inference_mode(), ctrl if ctrl is not None else nullcontext():
            with instrument(ctrl) if trace else nullcontext():
                torch.cuda.synchronize()
                if trace:
                    torch.cuda.cudart().cudaProfilerStart()
                    torch.cuda.nvtx.range_push("asi.chunk.generate")
                try:
                    out = model.generate_samples_from_batch(*positional, **kwargs)
                    torch.cuda.synchronize()
                finally:
                    if trace:
                        torch.cuda.nvtx.range_pop()
                        torch.cuda.cudart().cudaProfilerStop()
        summary = ctrl.finish() if ctrl is not None else {"strategy": "dense"}
        values = {
            "action": out["action"][0][service.cfg.history_length :, : service.cfg.action_dim].detach().cpu().clone(),
            "future_vision": out["vision"][0][:, :, 1:].detach().cpu().clone(),
        }
        values["action"][:, -1] = 1 - values["action"][:, -1]
        assert all(torch.isfinite(x).all() for x in values.values())
        return values, summary

    report = dict(
        mode=args.mode,
        server_source=str(source),
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=expected, text=True).strip(),
        capture=str(CAPTURE),
        capture_sha256=hashlib.sha256(CAPTURE.read_bytes()).hexdigest(),
        seed=capture["kwargs"]["seed"],
        steps=4,
        shift=5,
        guidance=3,
        compile=False,
        graph=False,
        warmup=5,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        source_sha256={
            str(f.relative_to(expected)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [source, expected / "cosmos_framework/model/generator/mot/unified_mot.py"]
        },
    )
    try:
        for i in range(5):
            reference, _ = run()
            print("WARMUP", args.mode, i + 1, flush=True)
        value, summary = run(True)
        assert all(torch.equal(value[k], reference[k]) for k in value)
        assert len(records) == 224
        assert len(score_calls) == (28 if args.mode == "asi" else 0)
        if args.mode == "asi":
            assert (summary["dense_stack_count"], summary["sparse_stack_count"]) == (1, 7)
            if old_eager is not None:
                assert all(torch.equal(value[k], old_eager[k]) for k in value)
                report["reference_eager"] = str(args.reference_eager)
                report["reference_eager_sha256"] = hashlib.sha256(args.reference_eager.read_bytes()).hexdigest()
                report["matches_previous_asi_eager_exact"] = True
            else:
                report["matches_previous_asi_eager_exact"] = None
        report.update(summary=summary, layers=records, score_calls=score_calls, trace_matches_warmup_exact=True)
        torch.save(value, args.output / "outputs.pt")
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print("COMPLETE", args.mode, args.output, flush=True)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
