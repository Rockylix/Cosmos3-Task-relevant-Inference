"""Observation-selected future pruning with branch-local early-exit hidden heads.

Plans are built once per chunk on step0 conditional and replayed layer-by-layer
on BOTH CFG branches of ALL steps. Full latent and UniPC histories are preserved.
"""

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.inference.future_instruction_attention import (
    compare_attention_output,
    instruction_attention_mass,
)
from cosmos_framework.inference.specprune_exit_plan import ExitConfig, ExitHidden, ObservationPlan
from cosmos_framework.inference.specprune_future import SpecPruneFuture, resolve_instruction_indices


class SpecPruneObservationExit(SpecPruneFuture):
    def __init__(self, model, config=ExitConfig()):
        self.model, self.net, self.config = model, model.net, config
        self.plan = ObservationPlan(config)
        self.chunk = 0

    def reset(self):
        self.plan.reset()
        self.chunk = 0

    def _capture(self, *, layer_index, q_gen, k_ar, k_gen, v_ar, v_gen, attn_output_gen, scaling):
        ids = self.active_patch_ids
        obs_ids = torch.where(ids < self.spatial)[0]
        if len(obs_ids) != self.spatial:
            raise AssertionError("L0 must stay complete")
        keys, values = torch.cat((k_ar, k_gen)), torch.cat((v_ar, v_gen))
        if layer_index in (0, 1, *self.config.global_blocks):
            score = instruction_attention_mass(q_gen[obs_ids], keys, values, self.instruction_indices, scale=scaling)
            check = compare_attention_output(score.output, attn_output_gen[obs_ids])
            self.current_scores[layer_index] = score.mass.detach().clone()
            if layer_index in self.config.global_blocks:
                self.plan.global_scores[layer_index] = score.mass.detach().clone()
        else:
            # GEN is [all current vision | all 33 actions]; exclude condition q0.
            action_ids = torch.arange(len(ids) + 1, len(ids) + 33, device=ids.device)
            q = q_gen[action_ids].float().transpose(0, 1)
            repeat = q.shape[0] // keys.shape[1]
            k = keys.float().repeat_interleave(repeat, dim=1).transpose(0, 1)
            v = values.float().repeat_interleave(repeat, dim=1).transpose(0, 1)
            probs = (q @ k.transpose(-1, -2) * scaling).softmax(-1)
            output = (probs @ v).transpose(0, 1)
            check = compare_attention_output(output, attn_output_gen[action_ids])
            obs_probability = probs[:, :, len(k_ar) + obs_ids].mean(0)
            self.plan.update(layer_index, obs_probability)
        if check["relative_l2"] > 0.02 or check["cosine"] < 0.999:
            raise AssertionError(f"Attention validation failed B{layer_index}: {check}")
        self.validation.append(dict(block=layer_index, **check))

    def _forward(self, original, patch_ids, patches, actions, timestep, step, branch, force_full):
        from cosmos_framework.data.generator.sequence_packing.runtime import (
            from_all_seq,
            from_und_gen_splits,
            get_all_seq,
            get_gen_seq,
            get_und_seq,
        )

        lm = self.net.language_model.model
        full_ids = patch_ids
        runtime, meta, packed, seq_ids = self._view(original, patch_ids, patches, actions, timestep)
        full_runtime, full_packed = runtime, packed
        stash = ExitHidden(get_all_seq(runtime))
        sentinel = torch.tensor([], dtype=patches.to(**self.model.tensor_kwargs).dtype, device=patches.device)
        pos = packed.position_ids
        cos, sin = lm.rotary_emb(sentinel, position_ids=pos.unsqueeze(0) if pos.ndim == 1 else pos.unsqueeze(1))
        cos, sin = cos.squeeze(0), sin.squeeze(0)
        rope = (from_all_seq(cos, runtime), from_all_seq(sin, runtime))
        building = step == 0 and branch == "conditional"
        for block, layer in enumerate(lm.layers):
            self.active_patch_ids = patch_ids
            capture = building and block in (0, 1, *self.config.global_blocks, *self.config.update_blocks)
            if layer.self_attn._attention_stats_capture_callback is not None:
                raise RuntimeError("Another attention capture is active")
            if capture:
                layer.self_attn._attention_stats_capture_callback = self._capture
            n_gen = len(get_gen_seq(runtime))
            try:
                runtime, metadata, kv = layer(
                    runtime, meta, rope, natten_metadata=None, memory_value=None, gen_only=False
                )
            finally:
                layer.self_attn._attention_stats_capture_callback = None
            if metadata or kv is not None or not torch.isfinite(get_all_seq(runtime)).all():
                raise AssertionError("Unexpected metadata or nonfinite block output")
            self.token_rows.append(
                dict(step=step, branch=branch, block=block, gen_tokens=n_gen, full_gen_tokens=9 * self.spatial + 33)
            )
            if building:
                if block in (0, 1):
                    self.plan.local(block, self.current_scores[block])
                elif block in self.config.prune_blocks:
                    self.plan.prune(block)
            if block not in self.plan.masks or force_full:
                continue
            mask = self.plan.masks[block]
            new_ids = torch.cat(
                (torch.arange(self.spatial, device=patch_ids.device), torch.where(mask.repeat(8))[0] + self.spatial)
            )
            if len(new_ids) == len(patch_ids):
                if not torch.equal(new_ids, patch_ids):
                    raise AssertionError("Non-nested mask")
                continue
            old_hidden = get_all_seq(runtime)
            new_runtime, new_meta, new_packed, new_seq = self._view(
                original, new_ids, patches[new_ids], actions, timestep
            )
            lookup = torch.full((original.sequence_length,), -1, device=patch_ids.device, dtype=torch.long)
            lookup[seq_ids] = torch.arange(len(seq_ids), device=patch_ids.device)
            relative = lookup[new_seq]
            if (relative < 0).any():
                raise AssertionError("Deleted hidden reinjected inside Transformer")
            retained = torch.zeros(len(seq_ids), dtype=torch.bool, device=patch_ids.device)
            retained[relative] = True
            stash.put(seq_ids[~retained], old_hidden[~retained])
            self.exit_rows.append(dict(step=step, branch=branch, block=block, dropped=int((~retained).sum())))
            runtime = from_all_seq(old_hidden[relative], new_runtime)
            cos, sin = cos[relative], sin[relative]
            rope = (from_all_seq(cos, runtime), from_all_seq(sin, runtime))
            meta, packed, seq_ids, patch_ids = new_meta, new_packed, new_seq, new_ids
        stash.put(seq_ids, get_all_seq(runtime))
        runtime = from_all_seq(stash.finish(), full_runtime)
        runtime = from_und_gen_splits(lm.norm(get_und_seq(runtime)), lm.norm_moe_gen(get_gen_seq(runtime)), runtime)
        hidden = get_all_seq(runtime)
        vision = self.net.llm2vae(hidden[full_packed.vision.sequence_indexes])
        vision[full_ids < self.spatial] = 0
        vision *= self.valid_patch_scalars.to(vision.dtype)
        decoded = {}
        self.net._decode_action(full_packed, hidden, decoded)
        action = decoded["preds_action"][0]
        action *= (1 - original.action.condition_mask[0]).to(action)
        if self.raw_action_dim is not None:
            action[:, self.raw_action_dim :] = 0
        if not torch.isfinite(vision).all() or not torch.isfinite(action).all():
            raise AssertionError("Nonfinite full output head")
        return torch.cat((vision.flatten(), action.flatten())), full_ids

    @torch.inference_mode()
    def generate(self, data_batch, *, seed, guidance=3.0, num_steps=4, shift=5.0, force_full=False, output_dir=None):
        from cosmos_framework.data.generator.action.action_processing import (
            ActionProcessor,
            get_action_processing_records,
        )
        from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler

        if len(seed) != 1 or num_steps != 4 or guidance != 3 or shift != 5:
            raise ValueError("This experiment requires batch1, four steps, guidance3, shift5")
        net = self.net
        observation_rgb = None
        self.observation_geometry = None
        if True:
            from cosmos_framework.inference.specprune_observation import condition_rgb

            observation_rgb = condition_rgb(data_batch, self.model.input_video_key)
        if net.config.joint_attn_implementation != "two_way" or net.video_temporal_causal or net.config.sound_gen:
            raise ValueError("Only original two-way Edge vision/action is supported")
        if net.num_hidden_layers != 28 or net.pad_for_cuda_graphs:
            raise ValueError("Expected 28 eager unpadded blocks")
        plans, clean, cond, uncond, initial, _, _, has_action = self.model._prepare_inference_data(
            data_batch, seed, False
        )
        if len(initial) != 1 or len(clean.x0_tokens_vision) != 1 or not has_action:
            raise ValueError("Expected one joint video/action sample")
        original_vision_shape = clean.x0_tokens_vision[0].shape
        shape = original_vision_shape
        if len(shape) == 5 and shape[0] == 1:
            shape = shape[1:]
        c, t, h, w = shape
        if t != 9:
            raise ValueError(f"Expected L0..L8, got {shape}")
        self.action_shape = clean.x0_tokens_action[0].shape
        if self.action_shape[0] != 33:
            raise ValueError("Expected condition action + 32 predicted actions")
        self.raw_action_dim = clean.raw_action_dim[0] if clean.raw_action_dim is not None else None
        p = net.latent_patch_size
        gh, gw = math.ceil(h / p), math.ceil(w / p)
        self.spatial = gh * gw
        self.grid = (8, gh, gw)
        if self.spatial != 340:
            raise ValueError("Confirmed budgets require 340 spatial tokens")
        vision_size = math.prod(shape)
        full_latent = initial[0][:vision_size].reshape(shape)
        actions = initial[0][vision_size:].reshape(self.action_shape)
        patches, _ = net.patchify_and_pack_latents([full_latent], [(9, gh, gw)])
        valid, _ = net.patchify_and_pack_latents([torch.ones_like(full_latent)], [(9, gh, gw)])
        self.valid_patch_scalars = valid.bool()
        patch_ids = torch.arange(len(patches), device=patches.device)
        obs_patches = None
        if observation_rgb is not None:
            from cosmos_framework.inference.specprune_observation import observation_patches

            if "image_size" not in data_batch or len(data_batch["image_size"]) != 1:
                raise ValueError("Observation Dynamic requires real image_size metadata")
            obs_patches, self.observation_geometry = observation_patches(
                observation_rgb,
                latent_hw=(h, w),
                latent_patch_size=p,
                spatial_factor=self.model.tokenizer_vision_gen.spatial_compression_factor,
                image_size=data_batch["image_size"][0],
            )
        self.exit_rows = []
        self.current_scores, self.validation, self.token_rows = {}, [], []
        self.mask = None
        caption = data_batch[self.model.input_caption_key][0]
        try:
            description = json.loads(caption)["actions"][0]["description"]
        except (ValueError, KeyError, TypeError):
            description = caption
        text_indices = resolve_instruction_indices(self.model.vlm_tokenizer, cond[0], description)
        self.instruction_indices = torch.tensor(text_indices, device=patches.device, dtype=torch.long)
        originals = {}
        for branch, tokens in [("conditional", cond), ("unconditional", uncond)]:
            packed = self.model._pack_input_sequence(
                plans,
                tokens,
                clean,
                torch.zeros((1, 1)),
                include_end_of_generation_token=self.model._derive_include_end_of_generation_token(),
            )
            packed.to_cuda()
            if packed.text_ids[: len(tokens[0])].tolist() != tokens[0]:
                raise ValueError("Packed instruction IDs do not match tokenized prefix")
            if packed.vision.token_shapes != [(9, gh, gw)]:
                raise ValueError("Native grid mismatch")
            originals[branch] = packed
        self.plan.begin(
            obs_patches.to(device=patches.device, dtype=torch.float32),
            originals["conditional"].sequence_length - 9 * self.spatial,
        )
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.model.sampler.cfg.num_train_timesteps,
            shift=self.model.sampler.cfg.shift,
            use_dynamic_shifting=self.model.sampler.cfg.use_dynamic_shifting,
        )
        scheduler.set_timesteps(num_steps, device=patches.device, shift=float(shift))
        rng = torch.Generator(device=patches.device).manual_seed(seed[0])
        torch.cuda.synchronize()
        begin = time.perf_counter()
        state = torch.cat((patches.flatten(), actions.flatten()))
        self.solver_rows = []
        for step, timestamp in enumerate(scheduler.timesteps):
            timestep = float(timestamp)
            dim = len(patch_ids) * net.patch_latent_dim
            patches, actions = state[:dim].reshape(len(patch_ids), -1), state[dim:].reshape(self.action_shape)
            conditional, new_ids = self._forward(
                originals["conditional"], patch_ids, patches, actions, timestep, step, "conditional", force_full
            )
            if step == 0:
                patches = patches[new_ids]
                patch_ids = new_ids
                state = torch.cat((patches.flatten(), actions.flatten()))
            elif not torch.equal(new_ids, patch_ids):
                raise AssertionError("Mask changed after first selection")
            unconditional, same_ids = self._forward(
                originals["unconditional"], patch_ids, patches, actions, timestep, step, "unconditional", force_full
            )
            if not torch.equal(same_ids, patch_ids) or conditional.shape != state.shape:
                raise AssertionError("CFG/solver coordinate mismatch")
            guided = unconditional + guidance * (conditional - unconditional)
            state = scheduler.step(
                model_output=guided, timestep=timestamp, sample=state.unsqueeze(0), return_dict=False, generator=rng
            )[0].squeeze(0)
            if not torch.isfinite(state).all():
                raise FloatingPointError("Nonfinite solver state")
            self.solver_rows.append(
                {
                    "step": step,
                    "timestep": timestep,
                    "state_scalars": state.numel(),
                    "patch_tokens": len(patch_ids),
                    "action_tokens": self.action_shape[0],
                }
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        final_patches = state[: len(patch_ids) * net.patch_latent_dim].reshape(len(patch_ids), -1)
        final_action = state[len(patch_ids) * net.patch_latent_dim :].reshape(self.action_shape)
        records = get_action_processing_records(data_batch)
        if not records or records[0] is None:
            raise ValueError("Missing original action externalization record")
        action_external = ActionProcessor.postprocess_action(final_action, records[0])
        self.plan.finish()
        self.mask = (torch.ones_like(self.plan.active) if force_full else self.plan.active).repeat(8).reshape(8, -1)
        for row in self.token_rows:
            candidates = [b for b in self.plan.masks if b < row["block"]]
            keep = 340 if force_full or not candidates else int(self.plan.masks[max(candidates)].sum())
            if row["gen_tokens"] != 373 + 8 * keep:
                raise AssertionError("Transformer input not consistent with shared spatial plan")
        if len(self.token_rows) != 224 or any(r["patch_tokens"] != 3060 for r in self.solver_rows):
            raise AssertionError("Full solver layout or forward count changed")
        self.selection_info = dict(
            source="observation",
            controller=False,
            dynamic_count=int(self.plan.dynamic_mask.sum()),
            global_count=int(self.plan.global_mask.sum()),
            first_chunk_local_only=self.chunk == 0,
            layers=self.plan.rows,
            early_exit_hidden=True,
            plan_source="step0_conditional_once_per_chunk",
        )
        info = {
            "chunk": self.chunk,
            "seed": seed[0],
            "instruction": description,
            "instruction_token_indices": text_indices,
            "instruction_tokens": self.model.vlm_tokenizer.convert_ids_to_tokens([cond[0][i] for i in text_indices]),
            "conditional_text_tokens": len(originals["conditional"].text_ids),
            "config": asdict(self.config),
            "selection": self.selection_info,
            "observation_geometry": self.observation_geometry,
            "selected_future_tokens": int(self.mask.sum()),
            "total_future_tokens": 8 * self.spatial,
            "mean_saved_gen_tokens_per_block_forward": float(
                np.mean([r["full_gen_tokens"] - r["gen_tokens"] for r in self.token_rows])
            ),
            "transformer_calls": len(self.token_rows),
            "solver_steps": self.solver_rows,
            "generation_with_scoring_s": elapsed,
            "attention_validation": self.validation,
            "exit_rows": self.exit_rows,
            "force_full": force_full,
        }
        self.last_info = info
        self.last_sparse_latent = final_patches.detach().cpu()
        self.last_patch_ids = patch_ids.detach().cpu()
        self.last_token_rows = self.token_rows
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_dir / "observation_scores.npz",
                rgb=observation_rgb.cpu().numpy(),
                dynamic=self.plan.dynamic_mask.cpu().numpy(),
                **{f"L0_instruction_B{b}": v.cpu().numpy() for b, v in self.current_scores.items()},
                **{f"action_L0_B{b}": v.cpu().numpy() for b, v in self.plan.action_scores.items()},
                **{f"mask_B{b}": v.cpu().numpy() for b, v in self.plan.masks.items()},
            )
            (output_dir / "metadata.json").write_text(json.dumps(info, indent=2, allow_nan=False))
        self.chunk += 1
        vision_latent = (
            final_patches.reshape(9, gh, gw, p, p, c)
            .permute(5, 0, 1, 3, 2, 4)
            .reshape(c, 9, gh * p, gw * p)[:, :, :h, :w]
        )
        # Preserve the native public shape, including its optional singleton batch.
        vision_latent = vision_latent.reshape(original_vision_shape)
        if output_dir is not None:
            torch.save({"vision": vision_latent.cpu(), "action": action_external.cpu()}, Path(output_dir) / "output.pt")
        return {"action": [action_external], "vision": [vision_latent]}
