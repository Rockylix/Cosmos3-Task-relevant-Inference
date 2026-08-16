# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboLab closed-loop server for Dense and V5.3 optimized A/C/D arms."""

from __future__ import annotations

import json
import socket
import statistics
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import tyro
from PIL import Image

from cosmos_framework.scripts.action_policy_server_robolab import (
    RobolabPolicyService,
    RobolabServerArgs,
    _extract_observation_image,
    _load_openpi_websocket_policy_server,
)
from cosmos_framework.scripts.action_policy_server_utils import get_local_ip
from cosmos_framework.scripts.robolab_v5_2_motion_core_stable_adaptive_velocity_cache import (
    DEFAULT_CORE_BLOCK_COUNT,
    DEFAULT_CORE_TOKEN_BUDGET,
    DEFAULT_K80_BUDGETS,
    DEFAULT_MAX_REPLACEMENTS,
    DEFAULT_REPLACEMENT_RELATIVE_THRESHOLD,
    DEFAULT_STABLE_BUDGETS,
    DEFAULT_STABLE_CV_PENALTY,
)
from cosmos_framework.scripts.robolab_v5_3_acd_packed_kernel_velocity_cache import (
    V53OptimizedACDController,
    V53VelocityCacheSampler,
)
from cosmos_framework.scripts.robolab_v5_3_c_cond_dense_step0 import (
    ALL_SPARSE_LATER_STRATEGY_VERSION,
    STRATEGY_VERSION,
    V53CConditionalDenseStep0AllSparseLaterController,
    V53CConditionalDenseStep0Controller,
)
from cosmos_framework.scripts.robolab_v6_core64_stable_fixed import (
    STABLE_BUDGETS as V6_STABLE_BUDGETS,
)
from cosmos_framework.scripts.robolab_v6_core64_stable_fixed import (
    STRATEGY_VERSION as V6_CORE64_STABLE_FIXED_STRATEGY_VERSION,
)
from cosmos_framework.scripts.robolab_v6_core64_stable_fixed import (
    TOKEN_BUDGETS as V6_TOKEN_BUDGETS,
)
from cosmos_framework.scripts.robolab_v6_core64_stable_fixed import (
    V6Core64StableFixedController,
)
from cosmos_framework.utils import log


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class V53ServerArgs(RobolabServerArgs):
    ablation_mode: Literal[
        "dense",
        "a",
        "c",
        "c_cond_step0",
        "c_cond_step0_b0_sparse",
        "c_core64_stable_fixed_b0_sparse",
        "d",
    ] = "c"
    """Closed-loop arm: Dense reference or one optimized V5.3 sparse arm."""

    roi_tokens_g1: int = DEFAULT_K80_BUDGETS[0]
    roi_tokens_g2: int = DEFAULT_K80_BUDGETS[1]
    roi_tokens_g3: int = DEFAULT_K80_BUDGETS[2]
    stable_tokens_g1: int = DEFAULT_STABLE_BUDGETS[0]
    stable_tokens_g2: int = DEFAULT_STABLE_BUDGETS[1]
    stable_tokens_g3: int = DEFAULT_STABLE_BUDGETS[2]
    core_block_count: int = DEFAULT_CORE_BLOCK_COUNT
    core_token_budget: int = DEFAULT_CORE_TOKEN_BUDGET
    stable_cv_penalty: float = DEFAULT_STABLE_CV_PENALTY
    replacement_relative_threshold: float = DEFAULT_REPLACEMENT_RELATIVE_THRESHOLD
    max_replacements: int = DEFAULT_MAX_REPLACEMENTS
    validate_intermediates: bool = False
    enable_nvtx: bool = False
    capture_first_mask_overlay_per_prompt: bool = False
    """Decode and save L1-L8 frames for the first request of each distinct prompt."""
    intervention_output_dir: Path = Path(
        "/root/robolab/experiments/preliminary/sparsity/velocity_cache/"
        "acd_packed_kernel_k80_1task_seed579362556_v1/server"
    )


class V53PolicyService(RobolabPolicyService):
    def _build_setup_args(self, args: RobolabServerArgs) -> Any:
        setup = super()._build_setup_args(args)
        updates = {"use_torch_compile": False, "use_cuda_graphs": False}
        if "guardrails" in type(setup).model_fields:
            updates["guardrails"] = False
        if "offload_guardrail_models" in type(setup).model_fields:
            updates["offload_guardrail_models"] = False
        return setup.model_copy(update=updates)

    def __init__(self, args: V53ServerArgs) -> None:
        if args.num_steps != 4 or float(args.shift) != 5.0:
            raise ValueError("V5.3 closed-loop evaluation requires num_steps=4 and shift=5")
        token_budgets = (int(args.roi_tokens_g1), int(args.roi_tokens_g2), int(args.roi_tokens_g3))
        stable_budgets = (
            int(args.stable_tokens_g1),
            int(args.stable_tokens_g2),
            int(args.stable_tokens_g3),
        )
        if str(args.ablation_mode) == "c_core64_stable_fixed_b0_sparse":
            token_budgets = V6_TOKEN_BUDGETS
            stable_budgets = V6_STABLE_BUDGETS
        if any(not 0 < value <= 340 for value in token_budgets) or not (
            token_budgets[0] >= token_budgets[1] >= token_budgets[2]
        ):
            raise ValueError("V5.3 token budgets must satisfy 340 >= G1 >= G2 >= G3 > 0")
        if any(value < 0 for value in stable_budgets) or not (
            stable_budgets[0] >= stable_budgets[1] >= stable_budgets[2]
        ):
            raise ValueError("V5.3 stable budgets must be non-negative and non-increasing")
        super().__init__(args)
        self._mode = str(args.ablation_mode)
        self._token_budgets = token_budgets
        self._stable_budgets = stable_budgets
        self._args = args
        self._output_dir = args.intervention_output_dir.expanduser().absolute()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._summaries: list[dict[str, Any]] = []
        self._capture_first_mask_overlay_per_prompt = bool(args.capture_first_mask_overlay_per_prompt)
        self._captured_overlay_prompts: set[str] = set()
        self._active_prompt: str | None = None
        self._active_conditioning_image: np.ndarray | None = None
        original_generate = self.model.generate_samples_from_batch

        def experimental_generate(*generate_args: Any, **generate_kwargs: Any) -> Any:
            request_index = len(self._summaries)
            request_dir = self._output_dir / f"request_{request_index:06d}"
            kwargs = dict(generate_kwargs)
            controller = None
            if self._mode != "dense":
                controller_cls = {
                    "c_cond_step0": V53CConditionalDenseStep0Controller,
                    "c_cond_step0_b0_sparse": V53CConditionalDenseStep0AllSparseLaterController,
                    "c_core64_stable_fixed_b0_sparse": V6Core64StableFixedController,
                }.get(self._mode, V53OptimizedACDController)
                controller = controller_cls(
                    ablation_mode=(
                        "c"
                        if self._mode in {"c_cond_step0", "c_cond_step0_b0_sparse", "c_core64_stable_fixed_b0_sparse"}
                        else self._mode
                    ),
                    torch=torch,
                    net=self.model.net,
                    guidance=float(kwargs.get("guidance", self.cfg.guidance)),
                    num_steps=int(kwargs.get("num_steps", self.cfg.num_steps)),
                    threshold=0.9,
                    token_budgets=self._token_budgets,
                    stable_budgets=self._stable_budgets,
                    core_block_count=int(args.core_block_count),
                    core_token_budget=int(args.core_token_budget),
                    stable_cv_penalty=float(args.stable_cv_penalty),
                    replacement_relative_threshold=float(args.replacement_relative_threshold),
                    max_replacements=int(args.max_replacements),
                    output_dir=request_dir,
                    validate_intermediates=bool(args.validate_intermediates),
                    enable_nvtx=bool(args.enable_nvtx),
                )
                kwargs["sampler"] = V53VelocityCacheSampler(self.model.sampler, controller)
            torch.cuda.synchronize()
            start = time.perf_counter()
            if controller is None:
                samples = original_generate(*generate_args, **kwargs)
            else:
                with controller:
                    samples = original_generate(*generate_args, **kwargs)
            torch.cuda.synchronize()
            wall_s = time.perf_counter() - start
            token_summary = controller.finish() if controller is not None else None
            prompt = self._active_prompt
            if (
                controller is not None
                and self._capture_first_mask_overlay_per_prompt
                and prompt is not None
                and prompt not in self._captured_overlay_prompts
            ):
                decoded = self.model.decode(samples["vision"][0]).detach().cpu().float()
                if decoded.ndim == 5 and int(decoded.shape[0]) == 1:
                    decoded = decoded[0]
                if decoded.ndim != 4 or int(decoded.shape[0]) != 3:
                    raise RuntimeError(f"Expected decoded [C,T,H,W], got {tuple(decoded.shape)}")
                if float(decoded.min()) < 0.0:
                    decoded = (decoded + 1.0) / 2.0
                frames = (
                    (decoded.clamp(0, 1) * 255)
                    .round()
                    .to(torch.uint8)
                    .permute(1, 2, 3, 0)
                    .numpy()
                )
                frames_dir = request_dir / "predicted_future_frames"
                frames_dir.mkdir(parents=True, exist_ok=True)
                for latent in range(1, 9):
                    decoded_index = latent * 4
                    Image.fromarray(np.ascontiguousarray(frames[decoded_index])).save(
                        frames_dir / f"frame_{decoded_index:03d}.png"
                    )
                if self._active_conditioning_image is not None:
                    Image.fromarray(np.ascontiguousarray(self._active_conditioning_image)).save(
                        request_dir / "conditioning_image.png"
                    )
                (request_dir / "mask_overlay_capture.json").write_text(
                    json.dumps(
                        {
                            "prompt": prompt,
                            "request_index": request_index,
                            "decoded_indices": [latent * 4 for latent in range(1, 9)],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                self._captured_overlay_prompts.add(prompt)
            self._summaries.append(
                {
                    "request_index": request_index,
                    "ablation_mode": self._mode,
                    "generation_wall_s": wall_s,
                    "token_savings": token_summary,
                }
            )
            warm = [float(item["generation_wall_s"]) for item in self._summaries[1:]]
            aggregate = {
                "schema_version": 1,
                "strategy_version": {
                    "dense": "dense",
                    "c_cond_step0": STRATEGY_VERSION,
                    "c_cond_step0_b0_sparse": ALL_SPARSE_LATER_STRATEGY_VERSION,
                    "c_core64_stable_fixed_b0_sparse": V6_CORE64_STABLE_FIXED_STRATEGY_VERSION,
                }.get(self._mode, "v5.3"),
                "ablation_mode": self._mode,
                "format_prompt_as_json": args.format_prompt_as_json,
                "compile": False,
                "cuda_graphs": False,
                "token_budgets": list(self._token_budgets),
                "completed_requests": len(self._summaries),
                "warm_request_count": len(warm),
                "warm_median_generation_s": statistics.median(warm) if warm else None,
                "warm_p90_generation_s": _percentile(warm, 0.9),
                "requests": self._summaries,
            }
            (self._output_dir / "aggregate.json").write_text(
                json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            log.info(f"[v5.3-closed-loop] mode={self._mode} request={request_index} wall_s={wall_s:.3f}")
            return samples

        self.model.generate_samples_from_batch = experimental_generate

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        if not self._capture_first_mask_overlay_per_prompt:
            return super().infer(obs)
        self._active_prompt = str(obs["prompt"])
        self._active_conditioning_image = _extract_observation_image(obs).copy()
        try:
            return super().infer(obs)
        finally:
            self._active_prompt = None
            self._active_conditioning_image = None


def serve(args: V53ServerArgs) -> None:
    displayed_budgets = (
        V6_TOKEN_BUDGETS
        if str(args.ablation_mode) == "c_core64_stable_fixed_b0_sparse"
        else (args.roi_tokens_g1, args.roi_tokens_g2, args.roi_tokens_g3)
    )
    log.info(
        f"[v5.3-closed-loop-server] host={socket.gethostname()} bind={args.host}:{int(args.port)} "
        f"mode={args.ablation_mode} shift={args.shift} "
        f"json_prompt={args.format_prompt_as_json} "
        f"budgets={displayed_budgets}"
    )
    service = V53PolicyService(args)
    local_ip = get_local_ip()
    log.info(f"[v5.3-closed-loop-server] accessible at ws://{local_ip}:{int(args.port)}/")
    server_cls = _load_openpi_websocket_policy_server()
    server_cls(policy=service, host=args.host, port=int(args.port), metadata={}).serve_forever()


def main() -> None:
    cascade = getattr(tyro.conf, "CascadeSubcommandArgs", tyro.conf.ConsolidateSubcommandArgs)
    args = tyro.cli(
        V53ServerArgs,
        description=__doc__,
        config=(tyro.conf.OmitArgPrefixes, cascade, tyro.conf.OmitSubcommandPrefixes),
    )
    serve(args)


if __name__ == "__main__":
    main()
