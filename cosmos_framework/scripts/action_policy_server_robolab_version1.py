# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboLab policy server for the canonical Cosmos3-Edge Version1 strategy."""

from cosmos_framework.inference.common.init import init_script

init_script()

import socket
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
    NUM_DENOISE_STEPS,
    STRATEGY_VERSION,
    TOKEN_BUDGET,
    Version1Controller,
)
from cosmos_framework.utils import log


class Version1ServerArgs(RobolabServerArgs):
    format_prompt_as_json: bool | None = True
    """Use the structured prompt format expected by the Edge policy checkpoint."""
    version1_output_dir: Path | None = None
    """Optional per-request selection artifacts; disabled by default."""


class Version1PolicyService(RobolabPolicyService):
    def _build_setup_args(self, args: RobolabServerArgs) -> Any:
        setup = super()._build_setup_args(args)
        updates = {"use_torch_compile": False, "use_cuda_graphs": False}
        return setup.model_copy(update=updates)

    def __init__(self, args: Version1ServerArgs) -> None:
        if int(args.num_steps) != NUM_DENOISE_STEPS or float(args.shift) != 5.0 or float(args.guidance) != 3.0:
            raise ValueError("The fixed Edge policy requires --guidance 3, --num-steps 4 and --shift 5")
        if args.format_prompt_as_json is not True:
            raise ValueError("Version1 Edge inference requires --format-prompt-as-json True")
        super().__init__(args)
        self._version1_output_dir = (
            args.version1_output_dir.expanduser().absolute() if args.version1_output_dir is not None else None
        )
        self._request_count = 0
        original_generate = self.model.generate_samples_from_batch

        def version1_generate(*generate_args: Any, **generate_kwargs: Any) -> Any:
            request_index = self._request_count
            controller = Version1Controller(
                torch=torch,
                net=self.model.net,
                guidance=float(generate_kwargs.get("guidance", self.cfg.guidance)),
                num_steps=int(generate_kwargs.get("num_steps", self.cfg.num_steps)),
                output_dir=(
                    self._version1_output_dir / f"request_{request_index:06d}"
                    if self._version1_output_dir is not None
                    else None
                ),
            )
            with controller:
                samples = original_generate(*generate_args, **generate_kwargs)
            summary = controller.finish()
            self._request_count += 1
            log.info(f"[robolab-version1] request={request_index} strategy={summary['strategy_version']}")
            return samples

        self.model.generate_samples_from_batch = version1_generate


def serve(args: Version1ServerArgs) -> None:
    log.info(
        f"[robolab-version1] host={socket.gethostname()} bind={args.host}:{int(args.port)} "
        f"strategy={STRATEGY_VERSION} budget=K{TOKEN_BUDGET}"
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
