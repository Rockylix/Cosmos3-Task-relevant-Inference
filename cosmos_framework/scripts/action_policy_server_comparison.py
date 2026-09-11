"""Read-only Dense / canonical ASI evaluation wrapper, selected by PYTHONPATH."""

from cosmos_framework.inference.common.init import init_script

init_script()

import copy
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq
from cosmos_framework.scripts.action_policy_server_robolab import (
    RobolabPolicyService,
    RobolabServerArgs,
    _load_openpi_websocket_policy_server,
)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


class ComparisonArgs(RobolabServerArgs):
    format_prompt_as_json: bool | None = True
    comparison_mode: Literal["dense", "asi"] = "dense"
    comparison_output: Path = Path("experiments/comparison/server")
    comparison_capture_chunk: int = 3


class ComparisonService(RobolabPolicyService):
    def _build_setup_args(self, args):
        return super()._build_setup_args(args).model_copy(update={"use_torch_compile": False, "use_cuda_graphs": False})

    def __init__(self, args):
        if (args.num_steps, args.guidance, args.shift, args.format_prompt_as_json) != (4, 3, 5, True):
            raise ValueError("Comparison requires 4 steps, guidance=3, shift=5, structured prompt")
        output = args.comparison_output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        if (output / "requests.jsonl").exists():
            raise FileExistsError(output)
        super().__init__(args)
        self._comparison_prompt = None
        self._comparison_chunk = 0
        self._comparison_request = 0
        original = self.model.generate_samples_from_batch
        import cosmos_framework.model.generator.mot.unified_mot as mot

        runtime = {
            "mode": args.comparison_mode,
            "model_source": mot.__file__,
            "wrapper_source": __file__,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "checkpoint": args.checkpoint_path,
            "policy_seed": args.seed,
            "deterministic_seed": args.deterministic_seed,
            "reset_seed_on_prompt": True,
            "compile": False,
            "cuda_graphs": False,
            "num_steps": args.num_steps,
            "shift": args.shift,
            "guidance": args.guidance,
            "capture_chunk": args.comparison_capture_chunk,
        }
        (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")

        def generate(*positional, **kwargs):
            capture = self._comparison_chunk == args.comparison_capture_chunk
            captured_input = cpu_tree({"args": positional, "kwargs": kwargs}) if capture else None
            finite = []
            handles = [
                layer.register_forward_hook(lambda m, a, o: finite.append(torch.isfinite(get_gen_seq(o[0])).all()))
                for layer in self.model.net.language_model.model.layers
            ]
            torch.cuda.synchronize()
            started = time.perf_counter()
            controller = None
            if args.comparison_mode == "asi":
                from cosmos_framework.scripts.robolab_version1 import Version1Controller

                controller = Version1Controller(torch=torch, net=self.model.net, guidance=3, num_steps=4)
            try:
                with controller if controller is not None else nullcontext():
                    samples = original(*positional, **kwargs)
                summary = controller.finish() if controller is not None else {}
            finally:
                for handle in handles:
                    handle.remove()
            if len(finite) != 224 or not torch.stack(finite).all().item():
                raise RuntimeError("Incomplete stack or nonfinite block output")
            for key in ("action", "vision"):
                if not torch.isfinite(samples[key][0]).all().item():
                    raise FloatingPointError(key)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            row = {
                "request": self._comparison_request,
                "prompt": self._comparison_prompt,
                "prompt_chunk": self._comparison_chunk,
                "seed": kwargs["seed"],
                "mode": args.comparison_mode,
                "generation_wall_s": elapsed,
                "block_calls": len(finite),
                "all_intermediate_finite": True,
                "captured": capture,
                "strategy_summary": summary,
            }
            with (output / "requests.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if capture:
                folder = output / "captures" / f"request_{self._comparison_request:06d}"
                folder.mkdir(parents=True)
                torch.save(
                    {
                        **captured_input,
                        "outputs": {k: samples[k][0].detach().cpu() for k in ("action", "vision")},
                        "metadata": row,
                    },
                    folder / "sample.pt",
                )
                (folder / "metadata.json").write_text(json.dumps(row, indent=2) + "\n")
            self._comparison_request += 1
            return samples

        self.model.generate_samples_from_batch = generate

    def infer(self, obs):
        if obs["prompt"] != self._comparison_prompt:
            self._comparison_prompt = obs["prompt"]
            self._comparison_chunk = 0
            self._rng = np.random.default_rng(self.cfg.seed)
        self._comparison_chunk += 1
        return super().infer(obs)


def main():
    from cosmos_framework.inference.common.args import tyro_cli

    args = tyro_cli(ComparisonArgs)
    service = ComparisonService(args)
    print(f"[comparison] READY {args.comparison_mode} port={args.port}", flush=True)
    _load_openpi_websocket_policy_server()(policy=service, host=args.host, port=args.port, metadata={}).serve_forever()


if __name__ == "__main__":
    main()
