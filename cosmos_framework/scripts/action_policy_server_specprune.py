"""Portable frozen ten-task SpecPrune server; optional decoder compilation, no paired Dense overhead."""

from typing import Literal

import numpy as np
import tyro

from cosmos_framework.inference.common.args import tyro_cli
from cosmos_framework.inference.specprune_exit_compile import compile_exit_layers
from cosmos_framework.inference.specprune_observation_exit import SpecPruneObservationExit
from cosmos_framework.scripts.action_policy_server_robolab import (
    RobolabPolicyService,
    RobolabServerArgs,
    _load_openpi_websocket_policy_server,
)


class SpecPruneServerArgs(RobolabServerArgs):
    specprune: bool = True
    specprune_backend: Literal["eager", "compile", "graph"] = "eager"


class SpecPrunePolicyService(RobolabPolicyService):
    def _build_setup_args(self, args):
        return super()._build_setup_args(args).model_copy(update={"use_torch_compile": False, "use_cuda_graphs": False})

    def __init__(self, args):
        if args.specprune and (args.num_steps, args.shift, args.guidance) != (4, 5, 3):
            raise ValueError("Frozen SpecPrune requires 4 steps, shift5, guidance3")
        if args.hidden_state_capture_dir or args.rope_qk_capture_dir or args.block_residual_capture_dir:
            raise ValueError("Use the experiment capture server; overlapping callbacks are unsupported")
        super().__init__(args)
        layers = None
        if args.specprune and args.specprune_backend != "eager":
            layers = compile_exit_layers(self.model, cuda_graphs=args.specprune_backend == "graph")
        self.adapter = SpecPruneObservationExit(self.model, compiled_layers=layers)
        self.prompt = None
        self.request_chunk = 0
        if args.specprune:
            self.model.generate_samples_from_batch = self.adapter.generate
        self.enabled = args.specprune

    def reset(self):
        self.adapter.reset()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.prompt = None
        self.request_chunk = 0

    def infer(self, obs):
        # Matches the measured ten-task protocol. For repeated identical prompts,
        # restart service per episode unless the caller explicitly invokes reset.
        if obs["prompt"] != self.prompt:
            self.reset()
            self.prompt = obs["prompt"]
        self.request_chunk += 1
        result = super().infer(obs)
        if self.enabled:
            info = self.adapter.last_info
            print(
                f"SPECPRUNE chunk={self.request_chunk} final_future={info['selected_future_tokens']}/2720 "
                f"saved_gen_per_block={info['mean_saved_gen_tokens_per_block_forward']:.2f}",
                flush=True,
            )
        return result


def main():
    args = tyro_cli(
        SpecPruneServerArgs,
        description=__doc__,
        config=(
            tyro.conf.OmitArgPrefixes,
            tyro.conf.ConsolidateSubcommandArgs,
            tyro.conf.OmitSubcommandPrefixes,
        ),
    )
    policy = SpecPrunePolicyService(args)
    _load_openpi_websocket_policy_server()(
        policy=policy, host=args.host, port=int(args.port), metadata={}
    ).serve_forever()


if __name__ == "__main__":
    main()
