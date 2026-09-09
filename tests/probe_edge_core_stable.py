"""One-GPU fixed-input regression probe for Edge policy cleanup (not a rollout)."""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--original", action="store_true")
args = p.parse_args()
ROOT = Path("/project/peilab/xiexinling")
sys.path.insert(0, str(args.repo))
from cosmos_framework.inference.common.init import init_script

init_script()
import numpy as np
import torch
from PIL import Image

from cosmos_framework.scripts.action_policy_server_robolab import (
    RobolabPolicyService,
    RobolabServerArgs,
    _build_data_batch_from_sample,
)
from cosmos_framework.scripts.robolab_version1 import Version1Controller


class Service(RobolabPolicyService):
    def _build_setup_args(self, config):
        setup = super()._build_setup_args(config)
        changes = dict(
            use_torch_compile=False,
            use_cuda_graphs=False,
            dp_replicate_size=1,
            dp_shard_size=1,
            cp_size=1,
            cfgp_size=1,
            guardrails=False,
            offload_guardrail_models=False,
        )
        return setup.model_copy(update={k: v for k, v in changes.items() if k in type(setup).model_fields})


vae = (
    ROOT
    / "assets/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
)
service = Service(
    RobolabServerArgs(
        checkpoint_path=str(ROOT / "assets/Cosmos3-Edge-Policy-DROID"),
        experiment_overrides=[
            f"model.config.tokenizer.vae_path={vae}",
            "model.config.tokenizer.object_store_credential_path_pretrained=",
            "model.config.tokenizer.bucket_name=",
        ],
        seed=0,
        deterministic_seed=False,
        guidance=3.0,
        num_steps=4,
        shift=5.0,
        format_prompt_as_json=True,
        decode_video=False,
        output_dir=args.output.parent / "model_output",
    )
)
result = []
for i, task in enumerate(["BananaInBowlTask", "ReorientAllMugsTask"]):
    path = ROOT / "runs/core_stable_dense_overlay_20260909/edge" / task / "chunk_01"
    prompt = json.loads((path / "mask.json").read_text())["prompt"]
    obs = {
        "observation/image": np.asarray(Image.open(path / "observation.png").convert("RGB")),
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": prompt,
    }
    batch = _build_data_batch_from_sample(service._build_sample(obs))
    calls = defaultdict(list)
    condition = []
    handles = []

    def trace_condition(module, pos, kw):
        packed = pos[0] if pos else kw["packed_seq"]
        if kw.get("und_only", False) or packed.vision is None:
            return
        v = packed.vision
        condition.append(
            {
                "hash": hashlib.sha256(v.tokens[0][:, 0].detach().float().cpu().numpy().tobytes()).hexdigest(),
                "noisy_frames": v.noisy_frame_indexes.tolist()
                if torch.is_tensor(v.noisy_frame_indexes)
                else str(v.noisy_frame_indexes),
            }
        )

    handles.append(service.model.net.register_forward_pre_hook(trace_condition, with_kwargs=True))
    for block, layer in enumerate(service.model.net.language_model.model.layers):
        for name in ["q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen"]:

            def trace(module, pos, output, key=f"{block}:{name}"):
                calls[key].append(int(pos[0].shape[0]))

            handles.append(getattr(layer.self_attn, name).register_forward_hook(trace))
    kwargs = dict(torch=torch, net=service.model.net, guidance=3.0, num_steps=4, output_dir=None)
    if args.original:
        kwargs.update(core_token_budget=80, stable_token_budget=104)
    controller = Version1Controller(**kwargs)
    with torch.inference_mode(), controller:
        output = service.model.generate_samples_from_batch(
            batch, guidance=3.0, num_steps=4, shift=5.0, seed=[[1826701615, 1367864807][i]]
        )
    summary = controller.finish()
    for handle in handles:
        handle.remove()
    assert len(calls) == 28 * 3 and all(v == [3093] + [1845] * 7 for v in calls.values()), dict(calls)
    assert len(condition) == 8 and len({v["hash"] for v in condition}) == 1, condition
    record = {
        "task": task,
        "summary": summary,
        "qkv_projection_calls": sum(map(len, calls.values())),
        "layers": 28,
        "stacks": 8,
        "tokens_per_stack": [3093] + [1845] * 7,
        "l0_input_unchanged_across_8_stacks": True,
        "l0_trace": condition,
        "outputs": {k: [v.detach().cpu() for v in output[k]] for k in ["action", "vision"]},
        "masks": {k: controller.plan[k].detach().cpu() for k in ["core_masks", "stable_mask", "execution_mask"]},
    }
    result.append(record)
    print(
        "PROBE_OK", task, "QKV calls=", record["qkv_projection_calls"], "L0 fixed input / live projections", flush=True
    )
torch.save(result, args.output)
