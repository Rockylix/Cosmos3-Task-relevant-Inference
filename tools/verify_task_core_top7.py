"""GPU functional replay: fixed task Core Top-7, current global Stable, native ASI velocity cache."""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    from cosmos_framework.inference.task_core_layers import TaskCoreLayerCache
    from cosmos_framework.scripts.asi_velocity_cache_chunk import CAPTURE, VAE, run_mode, write_json
    from cosmos_framework.scripts.asi_velocity_cache_smoke import Args, Service

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tasks = ["BananaInBowlTask", "BananaOnPlateTask"]
    sources = [
        "cosmos_framework/scripts/robolab_version1.py",
        "cosmos_framework/inference/task_core_layers.py",
        "cosmos_framework/inference/asi_velocity_cache.py",
        "cosmos_framework/scripts/asi_velocity_cache_smoke.py",
        "cosmos_framework/scripts/asi_velocity_cache_chunk.py",
        "tools/verify_task_core_top7.py",
    ]
    write_json(
        output / "manifest.json",
        {
            "tasks": tasks,
            "task_core_top7": True,
            "mode": "offline functional replay, not closed-loop",
            "inputs": {
                t: {
                    "path": str(CAPTURE.parent / f"{t}.pt"),
                    "sha256": hashlib.sha256((CAPTURE.parent / f"{t}.pt").read_bytes()).hexdigest(),
                }
                for t in tasks
            },
            "note": "Stored chunk3 inputs used to exercise request lifecycle; not real task chunk1 selections",
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "pythonpath": os.environ.get("PYTHONPATH"),
        },
    )
    service = Service(
        Args(
            eval_root=output,
            task_core_top7=True,
            checkpoint_path="/root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID",
            guardrails=False,
            output_dir=output / "model_output",
            experiment_overrides=[
                f"model.config.tokenizer.vae_path={VAE}",
                "model.config.tokenizer.object_store_credential_path_pretrained=",
                "model.config.tokenizer.bucket_name=",
            ],
        )
    )
    # Audit calls native generation, not the online logging wrapper with its own controller.
    service.model.generate_samples_from_batch = service.original_generate
    cache = TaskCoreLayerCache()
    rows = []
    for index, task in enumerate((tasks[0], tasks[0], tasks[1], tasks[0])):
        cache.begin_task(task)
        first = cache.blocks is None
        capture = torch.load(CAPTURE.parent / f"{task}.pt", map_location="cpu", weights_only=False)
        reference, _, selection, audit = run_mode(
            service.model, capture, "asi_cached_step0", core_block_count=7, fixed_core_blocks=cache.blocks
        )
        seed = capture["kwargs"]["seed"][0]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        original_topk = torch.topk
        calls = []

        def counted_topk(value, k, *positional, **kwargs):
            if tuple(value.shape) == (28,) and k == 7:
                calls.append(True)
            return original_topk(value, k, *positional, **kwargs)

        torch.topk = counted_topk
        try:
            samples, _, summary, controller, sampler = service.generate_fast(
                copy.deepcopy(capture["args"]), copy.deepcopy(capture["kwargs"]), layer_cache=cache
            )
        finally:
            torch.topk = original_topk
        assert len(calls) == (1 if first else 0)
        for key in ("action", "vision"):
            torch.testing.assert_close(samples[key][0].cpu(), reference[key], rtol=0, atol=0)
        torch.testing.assert_close(controller.plan["execution_mask"].cpu(), selection["execution_mask"], rtol=0, atol=0)
        assert summary["profiled_block_count"] == 28
        assert summary["stable_profile_blocks"] == list(range(28))
        assert summary["core_layers_reused"] == (not first)
        assert cache.blocks == tuple(summary["core_blocks"])
        assert sampler.cache.next_step == 4
        rows.append(
            {
                "replay_index": index + 1,
                "task": task,
                "cached_task_request_index": cache.completed_chunks,
                "captured_chunk": capture["metadata"]["chunk"],
                "seed": seed,
                "core_layer_topk_calls": len(calls),
                "fast_exact_audited_reference": True,
                "execution_mask_per_frame": controller.plan["execution_mask"].sum(1).cpu().tolist(),
                "summary": summary,
                "audit": audit,
            }
        )
        print(
            f"[task-top7] replay={index + 1} task={task} layers={cache.blocks} rank_calls={len(calls)} profiles=28 PASS",
            flush=True,
        )
    write_json(
        output / "verification.json",
        {
            "passed": True,
            "replay_requests": rows,
            "core_ranking_call_sequence": [r["core_layer_topk_calls"] for r in rows],
            "profile_count_sequence": [r["summary"]["profiled_block_count"] for r in rows],
            "no_performance_or_closed_loop_claim": True,
        },
    )
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
