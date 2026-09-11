"""Explicit ToCa-Future RoboLab server; native Dense remains unchanged."""

from cosmos_framework.inference.common.init import init_script

init_script()

import copy
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cosmos_framework.inference.toca_future import ToCaFutureConfig, ToCaFutureController
from cosmos_framework.scripts.action_policy_server_robolab import (
    RobolabPolicyService,
    RobolabServerArgs,
    _load_openpi_websocket_policy_server,
)
from cosmos_framework.utils import log


class ToCaServerArgs(RobolabServerArgs):
    format_prompt_as_json: bool | None = True
    toca_output_dir: Path = Path("experiments/toca_future_smoke/server")
    toca_full_steps: tuple[int, ...] = (0, 2)
    toca_fresh_ratio: float = 0.25
    toca_layer_slope: float = 0.5
    toca_period: int = 2
    toca_attention_backend: str = "reference"
    toca_validate_first_request: bool = True
    toca_reset_seed_on_prompt: bool = True


def tensor_metrics(reference, candidate):
    ref, value = reference.detach().float().flatten(), candidate.detach().float().flatten()
    if not bool((torch.isfinite(ref).all() & torch.isfinite(value).all()).item()):
        raise FloatingPointError("Nonfinite final output")
    delta = value - ref
    return {
        "mse": delta.square().mean().item(),
        "max_abs": delta.abs().max().item(),
        "relative_l2": (delta.norm() / ref.norm().clamp_min(1e-12)).item(),
        "cosine": torch.nn.functional.cosine_similarity(ref, value, dim=0).item(),
        "shape": list(reference.shape),
    }


class ToCaPolicyService(RobolabPolicyService):
    def _build_setup_args(self, args):
        return super()._build_setup_args(args).model_copy(update={"use_torch_compile": False, "use_cuda_graphs": False})

    def __init__(self, args: ToCaServerArgs):
        if args.num_steps != 4 or args.guidance != 3 or args.shift != 5 or args.format_prompt_as_json is not True:
            raise ValueError("The ToCa smoke requires four steps, CFG=3, shift=5 and structured prompt")
        self.toca_config = ToCaFutureConfig(
            full_steps=args.toca_full_steps,
            fresh_ratio=args.toca_fresh_ratio,
            layer_slope=args.toca_layer_slope,
            period=args.toca_period,
            attention_backend=args.toca_attention_backend,
        )
        self.output = args.toca_output_dir.expanduser().absolute()
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "requests.jsonl").exists():
            raise FileExistsError("Use a new ToCa output directory; refusing to overwrite a run")
        super().__init__(args)
        self._toca_request = 0
        self._toca_prompt = None
        self._toca_prompt_chunk = 0
        self._toca_reset_seed = args.toca_reset_seed_on_prompt
        original_generate = self.model.generate_samples_from_batch
        runtime = {
            "strategy": "toca-future",
            "config": asdict(self.toca_config),
            "checkpoint": args.checkpoint_path,
            "policy_seed": args.seed,
            "deterministic_seed": args.deterministic_seed,
            "reset_seed_on_prompt": self._toca_reset_seed,
            "compile": False,
            "cuda_graphs": False,
            "shift": args.shift,
            "guidance": args.guidance,
            "num_steps": args.num_steps,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "source": __file__,
        }
        (self.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")

        def generate(*positional: Any, **kwargs: Any):
            request = self._toca_request
            validate = args.toca_validate_first_request and request == 0
            dense = None
            if validate:
                # Validation is separate from the timed ToCa generation below.
                batch = positional[0] if positional else kwargs.get("data_batch")
                torch.save(
                    {"data_batch": copy.deepcopy(batch), "sampling_kwargs": kwargs}, self.output / "paired_input.pt"
                )
                dense = original_generate(*copy.deepcopy(positional), **copy.deepcopy(kwargs))
                # This remains an exact *reference instrumentation* check.
                # Joint-kernel score/AV numerical validation is performed by
                # benchmark_toca_joint; it is not claimed bitwise identical.
                observer = ToCaFutureController(
                    self.model.net,
                    replace(self.toca_config, full_steps=(0, 1, 2, 3), attention_backend="reference"),
                )
                with observer:
                    observed = original_generate(*copy.deepcopy(positional), **copy.deepcopy(kwargs))
                observer.finish()
                equivalence = {}
                for modality in ("action", "vision"):
                    equivalence[modality] = tensor_metrics(dense[modality][0], observed[modality][0])
                    if not torch.allclose(dense[modality][0], observed[modality][0], rtol=1e-4, atol=1e-4):
                        raise AssertionError(
                            f"All-full ToCa instrumentation changed Dense {modality}: {equivalence[modality]}"
                        )
                (self.output / "all_full_equivalence.json").write_text(json.dumps(equivalence, indent=2) + "\n")
                (self.output / "validation_scope.json").write_text(
                    json.dumps(
                        {
                            "all_full_equivalence_backend": "reference",
                            "actual_generation_backend": self.toca_config.attention_backend,
                            "joint_kernel_bitwise_equivalence_claimed": False,
                            "joint_numerical_validation_command": "python -m cosmos_framework.scripts.benchmark_toca_joint",
                        },
                        indent=2,
                    )
                    + "\n"
                )
                del observer, observed
            torch.cuda.synchronize()
            start = time.perf_counter()
            controller = ToCaFutureController(self.model.net, self.toca_config)
            traces, handles = {}, []
            if validate:
                for block, layer in enumerate(controller.layers):
                    modules = {name: getattr(layer.self_attn, f"{name}_proj_moe_gen") for name in ("q", "k", "v", "o")}
                    modules["mlp"] = layer.mlp_moe_gen
                    for name, module in modules.items():
                        key = f"{block}:{name}"
                        traces[key] = []
                        handles.append(
                            module.register_forward_pre_hook(
                                lambda m, inputs, key=key: traces[key].append(int(inputs[0].shape[0]))
                            )
                        )
            try:
                with controller:
                    samples = original_generate(*positional, **kwargs)
            finally:
                for handle in handles:
                    handle.remove()
            summary = controller.finish()
            if validate:
                for block in range(28):
                    records = [r for r in controller.records if r["block"] == block]
                    for name, field in (
                        ("q", "q_rows"),
                        ("k", "kv_rows"),
                        ("v", "kv_rows"),
                        ("o", "o_rows"),
                        ("mlp", "mlp_rows"),
                    ):
                        expected = [r[field] for r in records]
                        if traces[f"{block}:{name}"] != expected:
                            raise AssertionError(
                                f"Actual projection/MLP rows differ from ToCa schedule: {block}:{name}"
                            )
                (self.output / "actual_module_rows.json").write_text(json.dumps(traces, indent=2) + "\n")
            for modality in ("action", "vision"):
                if not torch.isfinite(samples[modality][0]).all().item():
                    raise FloatingPointError(f"Nonfinite final {modality}")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            summary.update(
                request=request,
                prompt=self._toca_prompt,
                prompt_chunk=self._toca_prompt_chunk,
                seed=kwargs.get("seed"),
                generation_wall_s=elapsed,
                validation_preceded=validate,
            )
            with (self.output / "requests.jsonl").open("a") as handle:
                handle.write(json.dumps(summary) + "\n")
            csv_path = self.output / "module_compute.csv"
            with csv_path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["request"] + list(controller.records[0]))
                if request == 0:
                    writer.writeheader()
                writer.writerows({"request": request, **record} for record in controller.records)
            if dense is not None:
                metrics = {
                    modality: tensor_metrics(dense[modality][0], samples[modality][0])
                    for modality in ("action", "vision")
                }
                (self.output / "paired_chunk_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
                torch.save(
                    {
                        "dense": {k: dense[k][0].cpu() for k in ("action", "vision")},
                        "toca_future": {k: samples[k][0].cpu() for k in ("action", "vision")},
                    },
                    self.output / "paired_outputs.pt",
                )
                log.info(f"[toca-validation] paired={metrics}")
            log.info(
                f"[toca-future] request={request} chunk={self._toca_prompt_chunk} generation={elapsed:.4f}s "
                f"Q_saved={summary['q']['saved_fraction']:.3%} MLP_saved={summary['mlp']['saved_fraction']:.3%}"
            )
            self._toca_request += 1
            return samples

        self.model.generate_samples_from_batch = generate

    def infer(self, obs):
        prompt = obs["prompt"]
        if prompt != self._toca_prompt:
            self._toca_prompt = prompt
            self._toca_prompt_chunk = 0
            if self._toca_reset_seed:
                self._rng = np.random.default_rng(self.cfg.seed)
        self._toca_prompt_chunk += 1
        return super().infer(obs)


def main():
    from cosmos_framework.inference.common.args import tyro_cli

    args = tyro_cli(ToCaServerArgs, description=__doc__)
    service = ToCaPolicyService(args)
    log.info(f"[toca-future] READY ws://{args.host}:{args.port}")
    _load_openpi_websocket_policy_server()(
        policy=service, host=args.host, port=args.port, metadata={"strategy": "toca-future"}
    ).serve_forever()


if __name__ == "__main__":
    main()
