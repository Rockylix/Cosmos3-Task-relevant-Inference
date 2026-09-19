"""Gate early-exit reconstruction against native Dense on the recorded chunk."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from probe_specprune_layout import CAPTURE, load_service

from cosmos_framework.inference.specprune_future import tensor_metrics
from cosmos_framework.inference.specprune_exit_metrics import future_vision
from cosmos_framework.inference.specprune_observation_exit import SpecPruneObservationExit

SOURCES = [
    "cosmos_framework/inference/specprune_exit_metrics.py",
    "cosmos_framework/inference/specprune_observation_exit.py",
    "cosmos_framework/inference/specprune_exit_plan.py",
    "cosmos_framework/inference/specprune_future.py",
    "cosmos_framework/inference/specprune_observation.py",
    "cosmos_framework/inference/future_instruction_attention.py",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = load_service().model
    record = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    batch = record["args"][0]
    kwargs = {k: record["kwargs"][k] for k in ("seed", "guidance", "num_steps", "shift")}
    adapter = SpecPruneObservationExit(model)
    with torch.inference_mode():
        ref = model.generate_samples_from_batch(copy.deepcopy(batch), **kwargs)
        full = adapter.generate(copy.deepcopy(batch), **kwargs, force_full=True)
        for key in ("vision", "action"):
            if full[key][0].shape != ref[key][0].shape:
                raise AssertionError(f"Native output contract mismatch: {key}")
        metrics = {k: tensor_metrics(full[k][0], ref[k][0]) for k in ("vision", "action")}
        print("FULL_KEEP", metrics, flush=True)
        if any(x["mse"] != 0 for x in metrics.values()):
            raise AssertionError("Full keep not bitwise identical; no rollouts")
        adapter.reset()
        sparse = adapter.generate(copy.deepcopy(batch), **kwargs, output_dir=args.output / "first")
        first = copy.deepcopy(adapter.last_info)
        error = {k: tensor_metrics(sparse[k][0], ref[k][0]) for k in ("vision", "action")}
        future_error = tensor_metrics(future_vision(sparse["vision"][0]), future_vision(ref["vision"][0]))
        second = adapter.generate(
            copy.deepcopy(batch), **{**kwargs, "seed": [kwargs["seed"][0] + 1]}, output_dir=args.output / "second"
        )
    if not adapter.last_info["exit_rows"]:
        raise AssertionError("No physical token pruning")
    root = Path(__file__).resolve().parents[1]
    result = dict(
        full_keep=metrics,
        sparse_error=error,
        sparse_future_error=future_error,
        native_shapes={k: list(ref[k][0].shape) for k in ("vision", "action")},
        first=first,
        second=adapter.last_info,
        dynamic_source="observation",
        dynamic_enabled=True,
        source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in SOURCES},
    )
    (args.output / "gate.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print("GATE PASS", json.dumps(error), flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
