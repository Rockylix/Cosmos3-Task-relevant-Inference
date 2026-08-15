# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboLab server for grouped temporal-closed ROI velocity-cache V5."""

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
from cosmos_framework.scripts.robolab_grouped_temporal_closed_roi_velocity_cache import (
    DEFAULT_BLOCK_GROUPS,
    VELOCITY_CACHE_STRATEGY_VERSION,
    GroupedTemporalClosedROISparseController,
    GroupedTemporalClosedVelocityCacheSampler,
)
from cosmos_framework.utils import log


class GroupedTemporalClosedROIServerArgs(RobolabServerArgs):
    roi_mass_threshold: float = 0.9
    """Per block-group/future-frame attention mass before temporal closure."""

    intervention_output_dir: Path = Path(
        "/root/robolab/experiments/preliminary/sparsity/velocity_cache/"
        "grouped_temporal_closed_roi_velocity_cache_shift5_v5/server"
    )
    """Request-level masks, velocity-cache traces, token counts, and timing."""


class GroupedTemporalClosedROIPolicyService(RobolabPolicyService):
    def _build_setup_args(self, args: RobolabServerArgs) -> Any:
        setup = super()._build_setup_args(args)
        updates = {"use_torch_compile": False, "use_cuda_graphs": False}
        if "guardrails" in type(setup).model_fields:
            updates["guardrails"] = False
        if "offload_guardrail_models" in type(setup).model_fields:
            updates["offload_guardrail_models"] = False
        return setup.model_copy(update=updates)

    def __init__(self, args: GroupedTemporalClosedROIServerArgs) -> None:
        if not 0.0 < args.roi_mass_threshold <= 1.0:
            raise ValueError("--roi-mass-threshold must be in (0,1]")
        if args.num_steps != 4:
            raise ValueError("V5 currently requires exactly four denoise steps")
        super().__init__(args)
        self._threshold = float(args.roi_mass_threshold)
        self._output_dir = args.intervention_output_dir.expanduser().absolute()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._summaries: list[dict[str, Any]] = []
        original_generate = self.model.generate_samples_from_batch

        def experimental_generate(*generate_args: Any, **generate_kwargs: Any) -> Any:
            request_index = len(self._summaries)
            request_dir = self._output_dir / f"request_{request_index:06d}"
            controller = GroupedTemporalClosedROISparseController(
                torch=torch,
                net=self.model.net,
                guidance=float(generate_kwargs.get("guidance", self.cfg.guidance)),
                num_steps=int(generate_kwargs.get("num_steps", self.cfg.num_steps)),
                threshold=self._threshold,
                block_groups=DEFAULT_BLOCK_GROUPS,
                output_dir=request_dir,
            )
            sampler = GroupedTemporalClosedVelocityCacheSampler(self.model.sampler, controller)
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
                "mean_saved_gen_tokens_per_sparse_block": sum(
                    item["token_savings"]["average_saved_gen_tokens_per_sparse_block"] for item in self._summaries
                )
                / len(self._summaries),
                "requests": self._summaries,
            }
            (self._output_dir / "aggregate.json").write_text(
                json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            log.info(
                "[v5-grouped-temporal-closed-roi] completed "
                f"request={request_index} wall_s={wall_s:.3f} "
                f"avg_saved={token_summary['average_saved_gen_tokens_per_sparse_block']:.1f}"
            )
            return samples

        self.model.generate_samples_from_batch = experimental_generate


def serve(args: GroupedTemporalClosedROIServerArgs) -> None:
    log.info(
        f"[v5-grouped-temporal-closed-roi-server] version={VELOCITY_CACHE_STRATEGY_VERSION} "
        f"host={socket.gethostname()} bind={args.host}:{int(args.port)} shift={args.shift} "
        f"threshold={args.roi_mass_threshold} block_groups={DEFAULT_BLOCK_GROUPS}"
    )
    service = GroupedTemporalClosedROIPolicyService(args)
    local_ip = get_local_ip()
    log.info(f"[v5-grouped-temporal-closed-roi-server] accessible at ws://{local_ip}:{int(args.port)}/")
    server_cls = _load_openpi_websocket_policy_server()
    server_cls(policy=service, host=args.host, port=int(args.port), metadata={}).serve_forever()


def main() -> None:
    cascade = getattr(tyro.conf, "CascadeSubcommandArgs", tyro.conf.ConsolidateSubcommandArgs)
    args = tyro.cli(
        GroupedTemporalClosedROIServerArgs,
        description=__doc__,
        config=(tyro.conf.OmitArgPrefixes, cascade, tyro.conf.OmitSubcommandPrefixes),
    )
    serve(args)


if __name__ == "__main__":
    main()
