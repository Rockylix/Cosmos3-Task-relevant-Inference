# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboLab policy server for the canonical Cosmos3-Edge Version1 strategy."""

from cosmos_framework.inference.common.init import init_script

init_script()

import json
import socket
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from cosmos_framework.scripts.action_policy_server_robolab import (
    RobolabPolicyService,
    RobolabServerArgs,
    _load_openpi_websocket_policy_server,
)
from cosmos_framework.scripts.action_policy_server_utils import get_local_ip
from cosmos_framework.scripts.robolab_version1 import (
    ACTION_HORIZON_WEIGHTS,
    NUM_DENOISE_STEPS,
    STRATEGY_VERSION,
    TOKEN_BUDGET,
    Version1Controller,
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


class Version1ServerArgs(RobolabServerArgs):
    format_prompt_as_json: bool | None = True
    """Use the structured prompt format expected by the Edge policy checkpoint."""
    version1_output_dir: Path = Path("/tmp/cosmos3_action_server/robolab_version1")
    """Per-request selection metadata and aggregate latency output."""


class Version1PolicyService(RobolabPolicyService):
    def _build_setup_args(self, args: RobolabServerArgs) -> Any:
        setup = super()._build_setup_args(args)
        updates = {"use_torch_compile": False, "use_cuda_graphs": False}
        return setup.model_copy(update=updates)

    def __init__(self, args: Version1ServerArgs) -> None:
        if int(args.num_steps) != NUM_DENOISE_STEPS or float(args.shift) != 5.0:
            raise ValueError("Version1 requires --num-steps 4 and --shift 5")
        if args.format_prompt_as_json is not True:
            raise ValueError("Version1 Edge inference requires --format-prompt-as-json True")
        super().__init__(args)
        self._version1_output_dir = args.version1_output_dir.expanduser().absolute()
        self._version1_output_dir.mkdir(parents=True, exist_ok=True)
        self._version1_summaries: list[dict[str, Any]] = []
        original_generate = self.model.generate_samples_from_batch

        def version1_generate(*generate_args: Any, **generate_kwargs: Any) -> Any:
            request_index = len(self._version1_summaries)
            controller = Version1Controller(
                torch=torch,
                net=self.model.net,
                guidance=float(generate_kwargs.get("guidance", self.cfg.guidance)),
                num_steps=int(generate_kwargs.get("num_steps", self.cfg.num_steps)),
                output_dir=self._version1_output_dir / f"request_{request_index:06d}",
            )
            torch.cuda.synchronize()
            start = time.perf_counter()
            with controller:
                # Keep the checkpoint's baseline sampler untouched.  In
                # particular, no cross-step velocity wrapper is installed.
                samples = original_generate(*generate_args, **generate_kwargs)
            torch.cuda.synchronize()
            wall_s = time.perf_counter() - start
            summary = controller.finish()
            self._version1_summaries.append(
                {
                    "request_index": request_index,
                    "generation_wall_s": wall_s,
                    "selection": summary,
                }
            )
            warm = [float(item["generation_wall_s"]) for item in self._version1_summaries[1:]]
            aggregate = {
                "schema_version": 1,
                "strategy_version": STRATEGY_VERSION,
                "format_prompt_as_json": True,
                "token_budget": TOKEN_BUDGET,
                "action_horizon_weights": list(ACTION_HORIZON_WEIGHTS),
                "velocity_cache_enabled": False,
                "completed_requests": len(self._version1_summaries),
                "warm_request_count": len(warm),
                "warm_median_generation_s": statistics.median(warm) if warm else None,
                "warm_p90_generation_s": _percentile(warm, 0.9),
                "requests": self._version1_summaries,
            }
            (self._version1_output_dir / "aggregate.json").write_text(
                json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            log.info(f"[robolab-version1] request={request_index} wall_s={wall_s:.3f}")
            return samples

        self.model.generate_samples_from_batch = version1_generate


def serve(args: Version1ServerArgs) -> None:
    log.info(
        f"[robolab-version1] host={socket.gethostname()} bind={args.host}:{int(args.port)} "
        f"strategy={STRATEGY_VERSION} budget=K{TOKEN_BUDGET} velocity_cache=False"
    )
    service = Version1PolicyService(args)
    local_ip = get_local_ip()
    log.info(f"[robolab-version1] accessible at ws://{local_ip}:{int(args.port)}/")
    server_cls = _load_openpi_websocket_policy_server()
    server_cls(policy=service, host=args.host, port=int(args.port), metadata={}).serve_forever()


def main() -> None:
    from cosmos_framework.inference.common.args import tyro_cli

    args = tyro_cli(Version1ServerArgs, description=__doc__)
    serve(args)


if __name__ == "__main__":
    main()
