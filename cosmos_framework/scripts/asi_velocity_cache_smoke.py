"""One dense / seven sparse ASI with request-local guided velocity reuse."""

from cosmos_framework.inference.common.init import init_script

init_script()

import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.inference.asi_velocity_cache import CacheSamplerAdapter, full_velocity_keep_mask
from cosmos_framework.inference.task_core_layers import TaskCoreLayerCache
from cosmos_framework.scripts.action_policy_server_robolab import (
    RobolabPolicyService,
    RobolabServerArgs,
    _load_openpi_websocket_policy_server,
)
from cosmos_framework.scripts.asi_velocity_cache_chunk import CAPTURE, metrics, run_mode, write_json
from cosmos_framework.scripts.robolab_version1 import Version1Controller
from cosmos_framework.utils import log


class Args(RobolabServerArgs):
    format_prompt_as_json: bool | None = True
    eval_root: Path
    gate_capture: Path = CAPTURE
    task_core_top7: bool = False
    """Fix task chunk1's Top-7 Core source layers; Stable still profiles all 28 layers each chunk."""


def append_csv(path, row):
    new = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


class Service(RobolabPolicyService):
    def _build_setup_args(self, args):
        return super()._build_setup_args(args).model_copy(update={"use_torch_compile": False, "use_cuda_graphs": False})

    def __init__(self, args):
        if (args.num_steps, args.shift, args.guidance, args.seed, args.deterministic_seed) != (4, 5, 3, 0, False):
            raise ValueError("Frozen protocol: steps4, shift5, CFG3, policyseed0, deterministic_seed=False")
        self.run = args.eval_root.resolve()
        self.manifest = json.loads((self.run / "manifest.json").read_text())
        if bool(self.manifest.get("task_core_top7", False)) != args.task_core_top7:
            raise ValueError("Manifest and --task-core-top7 disagree")
        metadata = json.loads(Path("/root/robolab/RoboLab/robolab/tasks/_metadata/task_metadata.json").read_text())
        self.task_for_prompt = {
            row["instruction"]: row["task_name"] for row in metadata if row["task_name"] in self.manifest["tasks"]
        }
        self.prompt, self.chunk = None, 0
        self.task_core_cache = TaskCoreLayerCache() if args.task_core_top7 else None
        if (self.run / "requests.jsonl").exists():
            raise FileExistsError("Refusing to append a second server run to existing requests")
        super().__init__(args)
        self.original_generate = self.model.generate_samples_from_batch
        if self.model.config.rectified_flow_inference_config.scheduler_type != "unipc":
            raise RuntimeError("Expected native UniPC")
        self.gate(torch.load(args.gate_capture, weights_only=False, map_location="cpu"))
        self.model.generate_samples_from_batch = self.generate_logged

    @torch.inference_mode()
    def generate_fast(self, positional, kwargs, *, cache_enabled=True, layer_cache=None):
        """No per-block auditing, per-step CPU copies, or artifact writes in the timed path."""
        torch.cuda.synchronize()
        started = time.perf_counter()
        controller = Version1Controller(
            torch=torch,
            net=self.model.net,
            guidance=3,
            num_steps=4,
            dense_step0=False,
            core_block_count=6 if layer_cache is None else layer_cache.count,
            fixed_core_blocks=None if layer_cache is None else layer_cache.blocks,
        )
        layout, handle, sampler = {}, None, None

        def capture_layout(module, args, keywords):
            if layout or keywords.get("und_only", args[2] if len(args) > 2 else False):
                return
            packed = args[0] if args else keywords["packed_seq"]
            layout.update(
                vision_shape=tuple(packed.vision.tokens[0].shape), grid_thw=tuple(packed.vision.token_shapes[0])
            )

        call_kwargs = dict(kwargs)
        if cache_enabled:
            handle = self.model.net.register_forward_pre_hook(capture_layout, with_kwargs=True)
            sampler = CacheSamplerAdapter(
                self.model.sampler,
                controller,
                layout,
                int(self.model.config.diffusion_expert_config.patch_spatial),
                True,
                step0_source="asi_cfg",
                audit=False,
            )
            call_kwargs["sampler"] = sampler
        try:
            with controller:
                samples = self.original_generate(*positional, **call_kwargs)
            summary = controller.finish()
        finally:
            if handle is not None:
                handle.remove()
        if cache_enabled and (sampler.evaluations != 4 or sampler.cache.next_step != 4):
            raise RuntimeError("Cache did not execute all four steps")
        if not all(torch.isfinite(samples[key][0]).all().item() for key in ("action", "vision")):
            raise FloatingPointError("Nonfinite action/vision")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if layer_cache is not None:
            layer_cache.commit(summary["core_blocks"])
            summary["task_layer_reference_chunk"] = 1
            summary["task_core_completed_chunks"] = layer_cache.completed_chunks
        return samples, elapsed, summary, controller, sampler

    def gate(self, capture):
        ordinary, initial, selection, ordinary_audit = run_mode(self.model, capture, "asi")
        cached, cached_initial, cached_selection, cached_audit = run_mode(self.model, capture, "asi_cached_step0")
        for key in ("action", "vision"):
            torch.testing.assert_close(ordinary[key], capture["outputs"][key], rtol=0, atol=0)
        torch.testing.assert_close(initial, cached_initial, rtol=0, atol=0)
        torch.testing.assert_close(selection["execution_mask"], cached_selection["execution_mask"], rtol=0, atol=0)
        seed = capture["kwargs"]["seed"][0]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        fast, _, summary, _, _ = self.generate_fast(copy.deepcopy(capture["args"]), copy.deepcopy(capture["kwargs"]))
        for key in ("action", "vision"):
            torch.testing.assert_close(fast[key][0].cpu(), cached[key], rtol=0, atol=0)
        action_equal = torch.equal(cached["action"], ordinary["action"])
        shape = cached_audit["layout"]["vision_shape"]
        keep = full_velocity_keep_mask(selection["execution_mask"], shape, (9, 17, 20), 2, initial.numel())
        n = cached["vision"].numel()
        kept_equal = torch.equal(cached["vision"].flatten()[keep[:n]], ordinary["vision"].flatten()[keep[:n]])
        write_json(
            self.run / "gpu_gate.json",
            {
                "passed": True,
                "ordinary_asi_exact_original_capture": True,
                "cached_step0_velocity_exact_asi": True,
                "fast_exact_audited_cache_path": True,
                "same_mask": True,
                "action_exact_asi_on_gate_chunk": action_equal,
                "retained_latent_exact_asi_on_gate_chunk": kept_equal,
                "cache_vs_asi": {key: metrics(cached[key], ordinary[key]) for key in cached},
                "ordinary_asi_audit": ordinary_audit,
                "cached_asi_audit": cached_audit,
                "controller": summary,
                "note": "Gate is offline; no gate request is counted as a closed-loop chunk or timed sample",
            },
        )
        torch.cuda.empty_cache()
        log.info("[asi-velocity] GPU gate PASS: one dense + seven sparse; fast output exactly matches audited output")

    def infer(self, obs):
        prompt = obs["prompt"]
        if prompt not in self.task_for_prompt:
            raise ValueError(f"Unexpected task: {prompt}")
        if prompt != self.prompt:
            self.prompt, self.chunk = prompt, 0
            self._rng = np.random.default_rng(self.cfg.seed)
            if self.task_core_cache is not None:
                self.task_core_cache.begin_task(self.task_for_prompt[prompt], reset=True)
        self.chunk += 1
        return super().infer(obs)

    def generate_logged(self, *positional, **kwargs):
        samples, elapsed, summary, controller, sampler = self.generate_fast(
            positional, kwargs, layer_cache=self.task_core_cache
        )
        # All D2H score reporting and file I/O happen after the measured generation.
        scores = {
            name: controller.plan[name].detach().cpu().tolist()
            for name in ("block_quality", "block_mass", "block_entropy")
        }
        core_weights = controller.plan["core_weights"].detach().cpu().tolist()
        task = self.task_for_prompt[self.prompt]
        row = {
            "task": task,
            "chunk": self.chunk,
            "seed": kwargs["seed"],
            "generation_s": elapsed,
            "all_final_finite": True,
            "velocity_source": "step0 ASI guided velocity (dense conditional + sparse unconditional)",
            "cached_velocity_scalars": sampler.cache.cache.numel(),
            "cache_evaluations": sampler.evaluations,
            "core_layer_weights": core_weights,
            **summary,
            **scores,
        }
        with (self.run / "requests.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        selected = summary["core_blocks"]
        append_csv(
            self.run / "chunk_topk_layers.csv",
            {
                "task": task,
                "chunk": self.chunk,
                "seed": kwargs["seed"][0],
                "generation_s": elapsed,
                **{
                    f"{'task_initial_rank' if self.task_core_cache is not None else 'rank'}_{i + 1}": block
                    for i, block in enumerate(selected)
                },
                "topk_sorted_by_block_id": " ".join(map(str, sorted(selected))),
                "core_layers_reused": summary["core_layers_reused"],
                "dense_forwards": summary["dense_stack_count"],
                "sparse_forwards": summary["sparse_stack_count"],
                "cached_velocity_scalars": row["cached_velocity_scalars"],
            },
        )
        for block in range(28):
            append_csv(
                self.run / "chunk_block_scores.csv",
                {
                    "task": task,
                    "chunk": self.chunk,
                    "block": block,
                    "R": scores["block_mass"][block],
                    "H": scores["block_entropy"][block],
                    "Q": scores["block_quality"][block],
                    "selected": block in selected,
                    "rank": selected.index(block) + 1 if block in selected else "",
                    "rank_scope": "task_first_chunk" if self.task_core_cache is not None else "current_chunk",
                    "current_core_weight": core_weights[selected.index(block)] if block in selected else "",
                },
            )
        log.info(
            f"[asi-velocity] {task} chunk={self.chunk} CoreTop{len(selected)}={selected} reused={summary['core_layers_reused']} profile=28 generation={elapsed:.4f}s D/S=1/7"
        )
        return samples


def main():
    from cosmos_framework.inference.common.args import tyro_cli

    args = tyro_cli(Args, description=__doc__)
    service = Service(args)
    log.info(f"[asi-velocity-server] READY {args.host}:{args.port}")
    _load_openpi_websocket_policy_server()(
        policy=service,
        host=args.host,
        port=args.port,
        metadata={
            "strategy": "asi_1d7s_velocity_cache",
            "step0_source": "asi_cfg",
            "task_core_top7": args.task_core_top7,
        },
    ).serve_forever()


if __name__ == "__main__":
    main()
