# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboLab server for the oracle Action-attention 90%-mass intervention."""

from __future__ import annotations

import json
import socket
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
from cosmos_framework.scripts.robolab_action_attention_mass90_intervention import (
    ActionAttentionMass90Controller,
)
from cosmos_framework.utils import log


class ActionMass90ServerArgs(RobolabServerArgs):
    action_attention_mass_threshold: float = 0.9
    """Within-frame mass retained for every predicted Action Query before union."""

    intervention_output_dir: Path = Path(
        "/root/robolab/cosmos-framework-edge-version1/experiments/preliminary/sparsity/action_attention_mass90/"
        "action_attention_mass90_sharedmask_BananaInBowlTask_sim_shift1_v1/server"
    )
    """Per-request token accounting output directory."""


class ActionMass90PolicyService(RobolabPolicyService):
    """Install one fresh controller around every generation request."""

    def _build_setup_args(self, args: RobolabServerArgs) -> Any:
        setup = super()._build_setup_args(args)
        updates = {"use_torch_compile": False, "use_cuda_graphs": False}
        if "guardrails" in type(setup).model_fields:
            updates["guardrails"] = False
        if "offload_guardrail_models" in type(setup).model_fields:
            updates["offload_guardrail_models"] = False
        return setup.model_copy(update=updates)

    def __init__(self, args: ActionMass90ServerArgs) -> None:
        if not 0.0 < args.action_attention_mass_threshold <= 1.0:
            raise ValueError("--action-attention-mass-threshold must be in (0,1]")
        super().__init__(args)
        self._mass_threshold = float(args.action_attention_mass_threshold)
        self._intervention_output_dir = args.intervention_output_dir.expanduser().absolute()
        self._intervention_output_dir.mkdir(parents=True, exist_ok=True)
        self._request_summaries: list[dict[str, Any]] = []
        dense_generate = self.model.generate_samples_from_batch

        def sparse_generate(*generate_args: Any, **generate_kwargs: Any) -> Any:
            request_index = len(self._request_summaries)
            request_dir = self._intervention_output_dir / f"request_{request_index:06d}"
            controller = ActionAttentionMass90Controller(
                torch=torch,
                net=self.model.net,
                guidance=float(generate_kwargs.get("guidance", self.cfg.guidance)),
                num_steps=int(generate_kwargs.get("num_steps", self.cfg.num_steps)),
                threshold=self._mass_threshold,
                output_dir=request_dir,
            )
            with controller:
                samples = dense_generate(*generate_args, **generate_kwargs)
            summary = {"request_index": request_index, **controller.finish()}
            self._request_summaries.append(summary)
            mean_saved = sum(item["mean_saved_gen_tokens"] for item in self._request_summaries) / len(
                self._request_summaries
            )
            aggregate = {
                "completed_requests": len(self._request_summaries),
                "mean_saved_gen_tokens_across_requests": mean_saved,
                "requests": self._request_summaries,
            }
            (self._intervention_output_dir / "aggregate_token_savings.json").write_text(
                json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            log.info(
                "[action-mass90] completed "
                f"request={request_index} mean_saved_gen_tokens={summary['mean_saved_gen_tokens']:.1f} "
                f"final_gen_tokens={summary['final_gen_tokens_by_stack']}"
            )
            return samples

        self.model.generate_samples_from_batch = sparse_generate


def serve(args: ActionMass90ServerArgs) -> None:
    log.info(
        f"[action-mass90-server] starting host={socket.gethostname()} bind={args.host}:{int(args.port)} "
        f"threshold={args.action_attention_mass_threshold}"
    )
    service = ActionMass90PolicyService(args)
    local_ip = get_local_ip()
    log.info(f"[action-mass90-server] accessible at ws://{local_ip}:{int(args.port)}/")
    server_cls = _load_openpi_websocket_policy_server()
    server_cls(policy=service, host=args.host, port=int(args.port), metadata={}).serve_forever()


def main() -> None:
    cascade = getattr(tyro.conf, "CascadeSubcommandArgs", tyro.conf.ConsolidateSubcommandArgs)
    args = tyro.cli(
        ActionMass90ServerArgs,
        description=__doc__,
        config=(tyro.conf.OmitArgPrefixes, cascade, tyro.conf.OmitSubcommandPrefixes),
    )
    serve(args)


if __name__ == "__main__":
    main()
