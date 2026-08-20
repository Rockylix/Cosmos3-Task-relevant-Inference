# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboLab server for step0-profile fixed-ROI/background-velocity-cache experiment."""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from typing import Any

import torch
import tyro

from cosmos_framework.scripts.action_policy_server_robolab import (
    RobolabPolicyService,
    RobolabServerArgs,
    _load_openpi_websocket_policy_server,
)
from cosmos_framework.scripts.action_policy_server_utils import get_local_ip
from cosmos_framework.scripts.robolab_step0_fixed_roi_velocity_cache import (
    GuidedBackgroundVelocityCacheSampler,
    Step0FixedROISparseController,
    VELOCITY_CACHE_STRATEGY_VERSION,
)
from cosmos_framework.utils import log


class Step0FixedROIServerArgs(RobolabServerArgs):
    fixed_roi_mass_threshold: float = 0.9
    """Mass retained after max aggregation of step-0 Action/Future profiles."""

    first_sparse_block: int = 4
    """First sparse Transformer block for denoise steps after step 0."""

    intervention_output_dir: Path = Path(
        "/root/robolab/cosmos-framework-edge-version1/experiments/preliminary/sparsity/velocity_cache/"
        "step0_profile_fixed_roi_velocity_cache_shift5_BananaInBowlTask_sim_v1/server"
    )
    """Request-level ROI, velocity-cache, token and timing artifacts."""


class Step0FixedROIPolicyService(RobolabPolicyService):
    def _build_setup_args(self, args: RobolabServerArgs) -> Any:
        setup = super()._build_setup_args(args)
        updates = {"use_torch_compile": False, "use_cuda_graphs": False}
        if "guardrails" in type(setup).model_fields:
            updates["guardrails"] = False
        if "offload_guardrail_models" in type(setup).model_fields:
            updates["offload_guardrail_models"] = False
        return setup.model_copy(update=updates)

    def __init__(self, args: Step0FixedROIServerArgs) -> None:
        if not 0.0 < args.fixed_roi_mass_threshold <= 1.0:
            raise ValueError("--fixed-roi-mass-threshold must be in (0,1]")
        super().__init__(args)
        self._threshold = float(args.fixed_roi_mass_threshold)
        self._first_sparse_block = int(args.first_sparse_block)
        self._output_dir = args.intervention_output_dir.expanduser().absolute()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._summaries: list[dict[str, Any]] = []
        original_generate = self.model.generate_samples_from_batch

        def experimental_generate(*generate_args: Any, **generate_kwargs: Any) -> Any:
            request_index = len(self._summaries)
            request_dir = self._output_dir / f"request_{request_index:06d}"
            controller = Step0FixedROISparseController(
                torch=torch,
                net=self.model.net,
                guidance=float(generate_kwargs.get("guidance", self.cfg.guidance)),
                num_steps=int(generate_kwargs.get("num_steps", self.cfg.num_steps)),
                threshold=self._threshold,
                first_sparse_block=self._first_sparse_block,
                output_dir=request_dir,
            )
            sampler = GuidedBackgroundVelocityCacheSampler(self.model.sampler, controller)
            kwargs = dict(generate_kwargs)
            kwargs["sampler"] = sampler
            torch.cuda.synchronize()
            start = time.perf_counter()
            with controller:
                samples = original_generate(*generate_args, **kwargs)
            torch.cuda.synchronize()
            wall_s = time.perf_counter() - start
            token_summary = controller.finish()
            sampler.save(request_dir)
            summary = {
                "request_index": request_index,
                "generation_wall_s": wall_s,
                "token_savings": token_summary,
            }
            self._summaries.append(summary)
            aggregate = {
                "strategy_version": VELOCITY_CACHE_STRATEGY_VERSION,
                "completed_requests": len(self._summaries),
                "mean_generation_wall_s": sum(item["generation_wall_s"] for item in self._summaries)
                / len(self._summaries),
                "mean_roi_tokens_per_future": sum(
                    item["token_savings"]["roi_tokens_per_future"] for item in self._summaries
                )
                / len(self._summaries),
                "mean_saved_gen_tokens_per_sparse_block": sum(
                    item["token_savings"]["saved_gen_tokens_per_sparse_block"] for item in self._summaries
                )
                / len(self._summaries),
                "requests": self._summaries,
            }
            (self._output_dir / "aggregate.json").write_text(
                json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            log.info(
                "[step0-fixed-roi] completed "
                f"request={request_index} wall_s={wall_s:.3f} "
                f"roi={token_summary['roi_tokens_per_future']}/340 "
                f"sparse_gen={token_summary['sparse_gen_tokens']}/{token_summary['full_gen_tokens']}"
            )
            return samples

        self.model.generate_samples_from_batch = experimental_generate


def serve(args: Step0FixedROIServerArgs) -> None:
    log.info(
        f"[step0-fixed-roi-server] version={VELOCITY_CACHE_STRATEGY_VERSION} "
        f"host={socket.gethostname()} bind={args.host}:{int(args.port)} "
        f"shift={args.shift} threshold={args.fixed_roi_mass_threshold} "
        f"first_sparse_block={args.first_sparse_block}"
    )
    service = Step0FixedROIPolicyService(args)
    local_ip = get_local_ip()
    log.info(f"[step0-fixed-roi-server] accessible at ws://{local_ip}:{int(args.port)}/")
    server_cls = _load_openpi_websocket_policy_server()
    server_cls(policy=service, host=args.host, port=int(args.port), metadata={}).serve_forever()


def main() -> None:
    cascade = getattr(tyro.conf, "CascadeSubcommandArgs", tyro.conf.ConsolidateSubcommandArgs)
    args = tyro.cli(
        Step0FixedROIServerArgs,
        description=__doc__,
        config=(tyro.conf.OmitArgPrefixes, cascade, tyro.conf.OmitSubcommandPrefixes),
    )
    serve(args)


if __name__ == "__main__":
    main()
