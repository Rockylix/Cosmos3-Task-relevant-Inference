# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboLab closed-loop server for Dense and V5.2 A-D K80 ablations."""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from typing import Any, Literal

import torch
import tyro

from cosmos_framework.scripts.action_policy_server_robolab import (
    RobolabPolicyService,
    RobolabServerArgs,
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
    V52MotionCoreStableAdaptiveController,
    V52VelocityCacheSampler,
)
from cosmos_framework.utils import log


class V52ServerArgs(RobolabServerArgs):
    ablation_mode: Literal["dense", "a", "b", "c", "d"] = "d"
    """Closed-loop arm: Dense reference or one V5.2 A-D ablation."""

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
    intervention_output_dir: Path = Path(
        "/root/robolab/cosmos-framework-edge-version1/experiments/preliminary/sparsity/velocity_cache/"
        "v5_2_motion_core_stable_adaptive_k80_5tasks_v1/server"
    )


class V52PolicyService(RobolabPolicyService):
    def _build_setup_args(self, args: RobolabServerArgs) -> Any:
        setup = super()._build_setup_args(args)
        updates = {"use_torch_compile": False, "use_cuda_graphs": False}
        if "guardrails" in type(setup).model_fields:
            updates["guardrails"] = False
        if "offload_guardrail_models" in type(setup).model_fields:
            updates["offload_guardrail_models"] = False
        return setup.model_copy(update=updates)

    def __init__(self, args: V52ServerArgs) -> None:
        if args.num_steps != 4 or float(args.shift) != 5.0:
            raise ValueError("V5.2 closed-loop evaluation requires num_steps=4 and shift=5")
        token_budgets = (int(args.roi_tokens_g1), int(args.roi_tokens_g2), int(args.roi_tokens_g3))
        stable_budgets = (
            int(args.stable_tokens_g1),
            int(args.stable_tokens_g2),
            int(args.stable_tokens_g3),
        )
        if any(not 0 < value <= 340 for value in token_budgets) or not (
            token_budgets[0] >= token_budgets[1] >= token_budgets[2]
        ):
            raise ValueError("V5.2 token budgets must satisfy 340 >= G1 >= G2 >= G3 > 0")
        if any(value < 0 for value in stable_budgets) or not (
            stable_budgets[0] >= stable_budgets[1] >= stable_budgets[2]
        ):
            raise ValueError("V5.2 stable budgets must be non-negative and non-increasing")
        super().__init__(args)
        self._mode = str(args.ablation_mode)
        self._token_budgets = token_budgets
        self._stable_budgets = stable_budgets
        self._args = args
        self._format_prompt_as_json = args.format_prompt_as_json
        self._output_dir = args.intervention_output_dir.expanduser().absolute()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._summaries: list[dict[str, Any]] = []
        original_generate = self.model.generate_samples_from_batch

        def experimental_generate(*generate_args: Any, **generate_kwargs: Any) -> Any:
            request_index = len(self._summaries)
            request_dir = self._output_dir / f"request_{request_index:06d}"
            kwargs = dict(generate_kwargs)
            controller = None
            sampler = None
            if self._mode != "dense":
                controller = V52MotionCoreStableAdaptiveController(
                    ablation_mode=self._mode,
                    torch=torch,
                    net=self.model.net,
                    guidance=float(kwargs.get("guidance", self.cfg.guidance)),
                    num_steps=int(kwargs.get("num_steps", self.cfg.num_steps)),
                    threshold=0.9,
                    token_budgets=self._token_budgets,
                    stable_budgets=self._stable_budgets,
                    core_block_count=int(self._args.core_block_count),
                    core_token_budget=int(self._args.core_token_budget),
                    stable_cv_penalty=float(self._args.stable_cv_penalty),
                    replacement_relative_threshold=float(self._args.replacement_relative_threshold),
                    max_replacements=int(self._args.max_replacements),
                    output_dir=request_dir,
                )
                sampler = V52VelocityCacheSampler(self.model.sampler, controller)
                kwargs["sampler"] = sampler
            torch.cuda.synchronize()
            start = time.perf_counter()
            if controller is None:
                samples = original_generate(*generate_args, **kwargs)
            else:
                with controller:
                    samples = original_generate(*generate_args, **kwargs)
            torch.cuda.synchronize()
            wall_s = time.perf_counter() - start
            token_summary = None
            if controller is not None and sampler is not None:
                token_summary = controller.finish()
                sampler.save(request_dir)
            summary = {
                "request_index": request_index,
                "ablation_mode": self._mode,
                "generation_wall_s": wall_s,
                "token_savings": token_summary,
            }
            self._summaries.append(summary)
            warm = [item["generation_wall_s"] for item in self._summaries[1:]]
            aggregate = {
                "schema_version": 1,
                "strategy_version": "dense" if self._mode == "dense" else "v5.2",
                "ablation_mode": self._mode,
                "format_prompt_as_json": self._format_prompt_as_json,
                "completed_requests": len(self._summaries),
                "mean_generation_wall_s": sum(item["generation_wall_s"] for item in self._summaries)
                / len(self._summaries),
                "warm_mean_generation_wall_s": sum(warm) / len(warm) if warm else None,
                "requests": self._summaries,
            }
            (self._output_dir / "aggregate.json").write_text(
                json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            log.info(f"[v5.2-closed-loop] mode={self._mode} request={request_index} wall_s={wall_s:.3f}")
            return samples

        self.model.generate_samples_from_batch = experimental_generate


def serve(args: V52ServerArgs) -> None:
    log.info(
        f"[v5.2-closed-loop-server] host={socket.gethostname()} bind={args.host}:{int(args.port)} "
        f"mode={args.ablation_mode} shift={args.shift} "
        f"json_prompt={args.format_prompt_as_json} "
        f"budgets={(args.roi_tokens_g1, args.roi_tokens_g2, args.roi_tokens_g3)}"
    )
    service = V52PolicyService(args)
    local_ip = get_local_ip()
    log.info(f"[v5.2-closed-loop-server] accessible at ws://{local_ip}:{int(args.port)}/")
    server_cls = _load_openpi_websocket_policy_server()
    server_cls(policy=service, host=args.host, port=int(args.port), metadata={}).serve_forever()


def main() -> None:
    cascade = getattr(tyro.conf, "CascadeSubcommandArgs", tyro.conf.ConsolidateSubcommandArgs)
    args = tyro.cli(
        V52ServerArgs,
        description=__doc__,
        config=(tyro.conf.OmitArgPrefixes, cascade, tyro.conf.OmitSubcommandPrefixes),
    )
    serve(args)


if __name__ == "__main__":
    main()
