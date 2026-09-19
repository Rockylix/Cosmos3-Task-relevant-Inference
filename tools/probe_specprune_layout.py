"""Read-only GPU layout probe on one existing recorded input; not a rollout."""

import json
from pathlib import Path

import torch

from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService, RobolabServerArgs

VAE = Path(
    "/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
)
CAPTURE = Path(
    "/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/experiments/asi_smoke10_s0_p0_v1/_temporary_inputs/BananaInBowlTask.pt"
)


class EagerService(RobolabPolicyService):
    def _build_setup_args(self, args):
        return super()._build_setup_args(args).model_copy(update={"use_torch_compile": False, "use_cuda_graphs": False})


def load_service():
    if not VAE.is_file():
        raise FileNotFoundError(VAE)
    return EagerService(
        RobolabServerArgs(
            checkpoint_path="/root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID",
            guardrails=False,
            format_prompt_as_json=True,
            decode_video=False,
            experiment_overrides=[
                f"model.config.tokenizer.vae_path={VAE}",
                "model.config.tokenizer.object_store_credential_path_pretrained=",
                "model.config.tokenizer.bucket_name=",
            ],
        )
    )


if __name__ == "__main__":
    service = load_service()
    record = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    batch = record["args"][0]
    with torch.inference_mode():
        plans, clean, cond, uncond, noise, _, _, has_action = service.model._prepare_inference_data(
            batch, record["kwargs"]["seed"], False
        )
        pack = service.model._pack_input_sequence(
            plans,
            cond,
            clean,
            torch.zeros((1, 1)),
            include_end_of_generation_token=service.model._derive_include_end_of_generation_token(),
        )
        pack.to_cuda()
    print(
        "LAYOUT",
        json.dumps(
            {
                "caption": batch["ai_caption"],
                "kwargs": record["kwargs"],
                "vision_shape": list(clean.x0_tokens_vision[0].shape),
                "action_shape": list(clean.x0_tokens_action[0].shape),
                "raw_action_dim": clean.raw_action_dim,
                "grid": pack.vision.token_shapes,
                "split_lens": pack.split_lens,
                "modes": pack.attn_modes,
                "text_indexes": pack.text_indexes.tolist(),
                "text_ids": pack.text_ids.tolist(),
                "cond_ids": cond,
                "uncond_ids": uncond,
                "decoded": service.model.vlm_tokenizer.decode(cond[0]),
                "tokens": service.model.vlm_tokenizer.convert_ids_to_tokens(cond[0]),
                "vision_noisy": [x.tolist() for x in pack.vision.noisy_frame_indexes],
                "action_noisy": [x.tolist() for x in pack.action.noisy_frame_indexes],
                "joint": service.model.net.config.joint_attn_implementation,
                "sampler": str(type(service.model.sampler)),
            },
            default=str,
        ),
        flush=True,
    )
