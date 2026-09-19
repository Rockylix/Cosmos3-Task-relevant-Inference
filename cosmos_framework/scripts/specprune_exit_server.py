"""Experiment-only RoboLab service; normal server and Baseline are untouched."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.inference.specprune_future import tensor_metrics
from cosmos_framework.inference.specprune_exit_metrics import future_vision
from cosmos_framework.inference.specprune_observation_exit import SpecPruneObservationExit
from cosmos_framework.scripts.action_policy_server_robolab import (
    RobolabPolicyService,
    RobolabServerArgs,
    _load_openpi_websocket_policy_server,
)

VAE = Path(
    "/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
)
TASKS = [
    "BananaInBowlTask",
    "BananaOnPlateTask",
    "ButterAboveRaisinTask",
    "BowlStackingLeftOnRightTask",
    "GrabABagelTask",
    "LargerObjectRaisinBoxInBinTask",
    "MustardInLeftBinTask",
    "RubiksCubeTask",
    "RubiksCubeLeftOfBowlTask",
    "MarkerInMugTask",
]


class Service(RobolabPolicyService):
    def _build_setup_args(self, args):
        return super()._build_setup_args(args).model_copy(update={"use_torch_compile": False, "use_cuda_graphs": False})

    def __init__(self, args, run, strategy, disable_dynamic=False, dynamic_source="observation"):
        self.run, self.strategy = run, strategy
        self.prompt = None
        self.chunk = 0
        metadata = json.loads(Path("/root/robolab/RoboLab/robolab/tasks/_metadata/task_metadata.json").read_text())
        self.task_for_prompt = {r["instruction"]: r["task_name"] for r in metadata if r["task_name"] in TASKS}
        super().__init__(args)
        self.original_generate = self.model.generate_samples_from_batch
        self.adapter = SpecPruneObservationExit(self.model)
        self.model.generate_samples_from_batch = self.generate_logged

    def infer(self, obs):
        if obs["prompt"] not in self.task_for_prompt:
            raise ValueError("Unexpected task prompt")
        if obs["prompt"] != self.prompt:
            self.prompt = obs["prompt"]
            self.chunk = 0
            self._rng = np.random.default_rng(self.cfg.seed)
            self.adapter.reset()
        self.chunk += 1
        return super().infer(obs)

    @torch.inference_mode()
    def generate_logged(self, batch, **kwargs):
        task = self.task_for_prompt[self.prompt]
        torch.cuda.synchronize()
        started = time.perf_counter()
        if self.strategy == "dense":
            samples = self.original_generate(batch, **kwargs)
            torch.cuda.synchronize()
            row = {"generation_s": time.perf_counter() - started}
        else:
            artifact = self.run / "artifacts" / task / f"chunk_{self.chunk:03d}"
            samples = self.adapter.generate(
                copy.deepcopy(batch), **kwargs, output_dir=artifact if self.chunk == 3 else None
            )
            torch.cuda.synchronize()
            row = copy.deepcopy(self.adapter.last_info)
            row["generation_including_capture_io_s"] = time.perf_counter() - started
            artifact.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                artifact / "raw_scores.npz",
                **{f"B{b}": score.float().cpu().numpy() for b, score in self.adapter.current_scores.items()},
                selected=self.adapter.mask.cpu().numpy(),
                dynamic=self.adapter.plan.dynamic_mask.cpu().numpy(),
            )
            (artifact / "token_rows.json").write_text(json.dumps(self.adapter.last_token_rows))
            # Independent matched-input Dense reference; NEVER fed back to the robot
            # or to selection/history, and excluded from sparse generation timing.
            reference = self.original_generate(copy.deepcopy(batch), **kwargs)
            row["paired_action_error"] = tensor_metrics(samples["action"][0][1:, :8], reference["action"][0][1:, :8])
            row["paired_vision_error"] = tensor_metrics(
                future_vision(samples["vision"][0]), future_vision(reference["vision"][0])
            )
            row["paired_action_horizon_mse"] = (
                (samples["action"][0][1:, :8] - reference["action"][0][1:, :8]).float().square().mean(-1).cpu().tolist()
            )
            torch.save(
                {"sparse": samples["action"][0].cpu(), "dense_same_input": reference["action"][0].cpu()},
                artifact / "paired_actions.pt",
            )
        row.update(task=task, chunk=self.chunk, request_seed=kwargs["seed"][0], strategy=self.strategy)
        with (self.run / "requests.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print("CHUNK", task, self.chunk, "future_kept", row.get("selected_future_tokens", 2720), flush=True)
        return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["dense", "specprune"], required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8041)
    parser.add_argument("--disable-dynamic", action="store_true")
    parser.add_argument("--dynamic-source", choices=["initial_noise", "observation"], default="observation")
    options = parser.parse_args()
    if options.disable_dynamic or options.dynamic_source != "observation":
        raise ValueError("This frozen experiment requires observation Dynamic")
    if (options.run / "requests.jsonl").exists():
        raise FileExistsError("Refusing to overwrite an earlier server run")
    args = RobolabServerArgs(
        checkpoint_path="/root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID",
        guardrails=False,
        format_prompt_as_json=True,
        decode_video=False,
        host="127.0.0.1",
        port=options.port,
        seed=0,
        deterministic_seed=False,
        guidance=3.0,
        num_steps=4,
        shift=5.0,
        output_dir=options.run / "server_outputs",
        experiment_overrides=[
            f"model.config.tokenizer.vae_path={VAE}",
            "model.config.tokenizer.object_store_credential_path_pretrained=",
            "model.config.tokenizer.bucket_name=",
        ],
    )
    service = Service(args, options.run, options.strategy, options.disable_dynamic, options.dynamic_source)
    print("SPECPRUNE_SERVICE_READY", flush=True)
    _load_openpi_websocket_policy_server()(policy=service, host=args.host, port=args.port, metadata={}).serve_forever()


if __name__ == "__main__":
    main()
