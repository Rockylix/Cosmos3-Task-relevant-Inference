"""Opt-in future-domain static SpecPrune transfer. No feature restoration.

The default model/deployment path never imports or installs this adapter.
Original position IDs, all condition/action tokens, and native UniPC equations
are preserved. First conditional B0/B1 are dense; one selection per chunk.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cosmos_framework.inference.future_instruction_attention import (
    capture_future_instruction_rows,
    export_heatmaps,
    scatter_future_grid,
)


@dataclass(frozen=True)
class SpecPruneConfig:
    local_k: int = 32
    global_k: int = 40
    global_blocks: tuple[int, int] = (13, 27)
    dynamic_threshold: float = 0.986
    low_change_cap: int = 313
    dynamic_enabled: bool = True
    dynamic_source: str = "initial_noise"

    def __post_init__(self):
        if self.dynamic_source not in ("initial_noise", "observation"):
            raise ValueError("Unknown Dynamic source")


def row_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Select only observed finite values; missing historical positions are unknown."""
    if scores.ndim != 2 or not 0 <= k <= scores.shape[1] or torch.isinf(scores).any():
        raise ValueError("Invalid scores or budget")
    valid = torch.isfinite(scores)
    order = torch.argsort(scores.nan_to_num(nan=-torch.inf), dim=-1, descending=True, stable=True)
    chosen = torch.zeros_like(valid)
    chosen.scatter_(1, order[:, :k], True)
    return chosen & valid


class ChunkSelector:
    def __init__(self, config: SpecPruneConfig = SpecPruneConfig()):
        self.config = config
        self.reset()

    def reset(self):
        self.previous_noise = None
        self.previous_global = {}
        self.pending_noise = None
        self.previous_observation = None
        self.pending_observation = None
        self.last_dynamic_mask = None
        self.last_similarities = None
        self.selected = None

    def begin(self, future_noise, *, observation=None):
        if self.pending_noise is not None:
            raise RuntimeError("Previous chunk was not completed/reset")
        if future_noise.ndim != 3 or future_noise.shape[0] != 8 or not torch.isfinite(future_noise).all():
            raise ValueError("Expected finite [8, spatial, patch_channels] initial future noise")
        self.pending_noise = future_noise.detach().clone()
        if self.config.dynamic_source == "observation":
            if observation is None or observation.ndim != 2 or observation.shape[0] != future_noise.shape[1]:
                raise ValueError("Missing or mismatched observation RGB patches")
            if not torch.isfinite(observation).all():
                raise ValueError("Nonfinite observation patches")
            self.pending_observation = observation.to(device=future_noise.device, dtype=torch.float32).detach().clone()
        self.selected = None

    def choose(self, local0, local1):
        if self.pending_noise is None or self.selected is not None:
            raise RuntimeError("Exactly one selection per chunk is allowed")
        shape = self.pending_noise.shape[:2]
        if local0.shape != shape or local1.shape != shape:
            raise ValueError("Local score layout changed")
        local = row_topk(local0, self.config.local_k) | row_topk(local1, self.config.local_k)
        global_mask = torch.zeros_like(local)
        dynamic = torch.zeros_like(local)
        similarities = None
        if self.previous_noise is not None:
            if self.previous_noise.shape != self.pending_noise.shape:
                raise ValueError("Layout changed; reset task history first")
            for block in self.config.global_blocks:
                if block not in self.previous_global:
                    raise RuntimeError("Missing historical Global layer")
                global_mask |= row_topk(self.previous_global[block], self.config.global_k)
            if self.config.dynamic_enabled:
                if self.config.dynamic_source == "observation":
                    current, previous = self.pending_observation, self.previous_observation
                    if previous is None or current.shape != previous.shape:
                        raise ValueError("Observation history mismatch; reset task first")
                    # Official RGB cosine denominator; do not center pixels or compare noise.
                    sim = (current * previous).sum(-1) / (current.norm(dim=-1) * previous.norm(dim=-1) + 1e-8)
                    similarities = sim.unsqueeze(0).expand(shape[0], -1)
                else:
                    similarities = F.cosine_similarity(self.pending_noise.float(), self.previous_noise.float(), dim=-1)
                low_change = similarities >= self.config.dynamic_threshold
                # Official mechanism protects the complement of top-capped stable patches.
                stable_scores = similarities.masked_fill(~low_change, torch.nan)
                stable = row_topk(stable_scores, min(self.config.low_change_cap, shape[1]))
                dynamic = ~stable
        self.selected = local | global_mask | dynamic
        self.last_dynamic_mask = dynamic.detach().clone()
        self.last_similarities = None if similarities is None else similarities.detach().clone()
        noise_similarity = similarities if self.config.dynamic_source == "initial_noise" else None
        obs_similarity = similarities if self.config.dynamic_source == "observation" else None
        return self.selected, {
            "dynamic_enabled": self.config.dynamic_enabled,
            "dynamic_source": self.config.dynamic_source,
            "local_counts": local.sum(-1).tolist(),
            "global_counts": global_mask.sum(-1).tolist(),
            "dynamic_counts": dynamic.sum(-1).tolist(),
            "union_counts": self.selected.sum(-1).tolist(),
            "initial_noise_similarity_mean": None if noise_similarity is None else float(noise_similarity.mean()),
            "initial_noise_similarity_max": None if noise_similarity is None else float(noise_similarity.max()),
            "observation_similarity_mean": None if obs_similarity is None else float(obs_similarity.mean()),
            "observation_similarity_min": None if obs_similarity is None else float(obs_similarity.min()),
            "observation_similarity_max": None if obs_similarity is None else float(obs_similarity.max()),
            "dynamic_shared_across_frames": bool((dynamic == dynamic[:1]).all()),
            "first_chunk_local_only": self.previous_noise is None,
        }

    def complete(self, global_scores):
        if self.pending_noise is None or self.selected is None:
            raise RuntimeError("Chunk selection is incomplete")
        for block in self.config.global_blocks:
            if block not in global_scores or global_scores[block].shape != self.selected.shape:
                raise ValueError("Missing/mismatched Global history")
        self.previous_noise = self.pending_noise
        self.previous_observation = self.pending_observation
        self.previous_global = {b: global_scores[b].detach().clone() for b in self.config.global_blocks}
        self.pending_noise = None
        self.pending_observation = None


def resolve_instruction_indices(tokenizer, cond_ids: list[int], instruction: str) -> list[int]:
    """Resolve exact body character span through the actual decoded/tokenized prompt.

    Offset mapping is accepted only if round-trip IDs exactly match the model
    input. Ambiguous/missing spans fail rather than using all condition keys.
    """
    rendered = tokenizer.decode(cond_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if not instruction or rendered.count(instruction) != 1:
        raise ValueError("Instruction body must occur exactly once in actual tokenized prompt")
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) != list(cond_ids):
        raise ValueError("Instruction offset mapping changed actual token IDs")
    start = rendered.index(instruction)
    end = start + len(instruction)
    result = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if b > start and a < end]
    if not result or any(cond_ids[i] in tokenizer.all_special_ids for i in result):
        raise ValueError("Invalid instruction span including special tokens")
    return result


def tensor_metrics(value, reference):
    a, b = value.double().flatten(), reference.double().flatten()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Metric shapes or finite check failed")
    return {
        "mse": float((a - b).square().mean()),
        "relative_l2": float((a - b).norm() / b.norm().clamp_min(1e-12)),
        "cosine": float(F.cosine_similarity(a, b, dim=0, eps=1e-12)),
        "max_absolute_error": float((a - b).abs().max()),
    }


class SpecPruneFuture:
    """Single-GPU, one 9-latent/33-action DROID sample, ordinary two-way attention.

    Deliberately separate from model.forward/generate_samples_from_batch. Uses
    the model's original layers, projections, positional embeddings, action
    processing and FlowUniPC scheduler; no monkey-patch of model computation.
    """

    def __init__(self, model, config=SpecPruneConfig()):
        self.model = model
        self.net = model.net
        self.config = config
        self.selector = ChunkSelector(config)
        self.chunk = 0

    def reset(self):
        self.selector.reset()
        self.chunk = 0

    def _view(self, original, patch_ids, patches, actions, timestep):
        """Construct actual smaller input metadata without reconstructing dropped data."""
        from cosmos_framework.model.generator.mot.attention import build_packed_sequence

        net = self.net
        packed = copy.copy(original)
        packed.vision = copy.copy(original.vision)
        packed.action = copy.copy(original.action)
        keep = torch.ones(original.sequence_length, dtype=torch.bool, device=patch_ids.device)
        keep[original.vision.sequence_indexes] = False
        keep[original.vision.sequence_indexes[patch_ids]] = True
        original_sequence_ids = torch.where(keep)[0]
        remap = torch.full((original.sequence_length,), -1, device=patch_ids.device, dtype=torch.long)
        remap[original_sequence_ids] = torch.arange(len(original_sequence_ids), device=patch_ids.device)
        packed.sequence_length = len(original_sequence_ids)
        packed.text_indexes = remap[original.text_indexes]
        packed.position_ids = original.position_ids.index_select(-1, original_sequence_ids)
        packed.sample_lens = [packed.sequence_length]
        splits, cursor = [], 0
        for count in original.split_lens:
            splits.append(int(keep[cursor : cursor + count].sum()))
            cursor += count
        packed.split_lens = splits
        packed.vision.sequence_indexes = remap[original.vision.sequence_indexes[patch_ids]]
        packed.action.sequence_indexes = remap[original.action.sequence_indexes]
        packed.action.mse_loss_indexes = remap[original.action.mse_loss_indexes]
        packed.action.tokens = [actions.to(**self.model.tensor_kwargs)]
        packed.action.timesteps = torch.full_like(original.action.timesteps, float(timestep))
        packed.vision.timesteps = torch.full_like(original.vision.timesteps, float(timestep))
        hidden, dtype = net._encode_text(packed)
        vision_hidden = net.vae2llm(patches.to(dtype))
        embeds = net._embed_packed_timesteps(packed.vision.timesteps.float() * net.timestep_scale, packed).to(dtype)
        frame_ids = torch.div(patch_ids, self.spatial, rounding_mode="floor")
        noisy = frame_ids > 0
        vision_hidden[noisy] += embeds[frame_ids[noisy] - 1]
        hidden[packed.vision.sequence_indexes] = vision_hidden
        net._encode_action(packed, hidden, dtype)
        gen_indices = torch.cat((packed.vision.sequence_indexes, packed.action.sequence_indexes))
        runtime, meta, natten = build_packed_sequence(
            net.config.joint_attn_implementation,
            packed_sequence=hidden,
            attn_modes=packed.attn_modes,
            split_lens=packed.split_lens,
            sample_lens=packed.sample_lens,
            packed_und_token_indexes=packed.text_indexes,
            packed_gen_token_indexes=gen_indices,
            num_heads=net.num_heads,
            head_dim=net.head_dim,
            num_layers=net.num_hidden_layers,
            token_shapes=packed.vision.token_shapes,
            natten_parameter_list=None,
            is_image_batch=packed.is_image_batch,
            cp_world_size=1,
            pad_for_cuda_graphs=False,
        )
        if natten is not None:
            raise ValueError("Neighborhood attention is not supported")
        return runtime, meta, packed, original_sequence_ids

    def _score_callback(self, **inputs):
        block = inputs.pop("layer_index")
        ids = self.active_patch_ids
        future_rows = torch.where(ids >= self.spatial)[0]
        result, validation = capture_future_instruction_rows(
            **inputs,
            future_query_indices=future_rows,
            instruction_ar_indices=self.instruction_indices,
            full_gen_attention_verified=True,
        )
        if validation["relative_l2"] > 0.02 or validation["cosine"] < 0.999:
            raise AssertionError(f"Attention kernel validation failed B{block}: {validation}")
        scores = torch.full((8 * self.spatial,), torch.nan, device=ids.device)
        scores[ids[future_rows] - self.spatial] = result.mass
        self.current_scores[block] = scores.reshape(8, self.spatial)
        self.validation.append({"block": block, **validation})

    def _forward(self, original, patch_ids, patches, actions, timestep, step, branch, force_full):
        from cosmos_framework.data.generator.sequence_packing.runtime import (
            from_all_seq,
            from_und_gen_splits,
            get_all_seq,
            get_gen_seq,
            get_und_seq,
        )

        lm = self.net.language_model.model
        runtime, meta, packed, original_seq_ids = self._view(original, patch_ids, patches, actions, timestep)
        sentinel = torch.tensor([], dtype=patches.to(**self.model.tensor_kwargs).dtype, device=patches.device)
        pos = packed.position_ids
        cos, sin = lm.rotary_emb(sentinel, position_ids=pos.unsqueeze(0) if pos.ndim == 1 else pos.unsqueeze(1))
        cos, sin = cos.squeeze(0), sin.squeeze(0)
        rope = (from_all_seq(cos, runtime), from_all_seq(sin, runtime))
        for block, layer in enumerate(lm.layers):
            self.active_patch_ids = patch_ids
            capture = step == 0 and branch == "conditional" and block in (0, 1, *self.config.global_blocks)
            old_callback = layer.self_attn._attention_stats_capture_callback
            if old_callback is not None:
                raise RuntimeError("Another attention collector is active")
            if capture:
                layer.self_attn._attention_stats_capture_callback = self._score_callback
            n_gen = len(get_gen_seq(runtime))
            try:
                runtime, metadata, kv = layer(
                    runtime, meta, rope, natten_metadata=None, memory_value=None, gen_only=False
                )
            finally:
                layer.self_attn._attention_stats_capture_callback = old_callback
            if metadata or kv is not None:
                raise ValueError("Unexpected MoE metadata or memory path")
            finite = torch.isfinite(get_gen_seq(runtime)).all()
            if not bool(finite):
                raise FloatingPointError(f"Nonfinite block output {step}/{branch}/B{block}")
            self.token_rows.append(
                {
                    "step": step,
                    "branch": branch,
                    "block": block,
                    "gen_tokens": n_gen,
                    "full_gen_tokens": 9 * self.spatial + self.action_shape[0],
                }
            )
            if step == 0 and branch == "conditional" and block == 1:
                selected, self.selection_info = self.selector.choose(self.current_scores[0], self.current_scores[1])
                if force_full:
                    selected = torch.ones_like(selected)
                    self.selector.selected = selected
                self.mask = selected.clone()
                new_ids = torch.cat(
                    (
                        torch.arange(self.spatial, device=patch_ids.device),
                        torch.where(selected.flatten())[0] + self.spatial,
                    )
                )
                # Cut once before any solver update/history exists, preserving original IDs.
                if len(new_ids) != len(patch_ids):
                    old_hidden = get_all_seq(runtime)
                    new_runtime, meta, new_packed, new_seq_ids = self._view(
                        original, new_ids, patches[new_ids], actions, timestep
                    )
                    # _view only builds input/layout here; discard its embedding values.
                    lookup = torch.full((original.sequence_length,), -1, device=patch_ids.device, dtype=torch.long)
                    lookup[original_seq_ids] = torch.arange(len(original_seq_ids), device=patch_ids.device)
                    relative = lookup[new_seq_ids]
                    if bool((relative < 0).any()):
                        raise AssertionError("Deleted token was re-injected")
                    runtime = from_all_seq(old_hidden.index_select(0, relative), new_runtime)
                    cos, sin = cos.index_select(0, relative), sin.index_select(0, relative)
                    rope = (from_all_seq(cos, runtime), from_all_seq(sin, runtime))
                    packed, original_seq_ids = new_packed, new_seq_ids
                    patch_ids = new_ids
        runtime = from_und_gen_splits(lm.norm(get_und_seq(runtime)), lm.norm_moe_gen(get_gen_seq(runtime)), runtime)
        hidden = get_all_seq(runtime)
        # Tokenwise original vision head; never unpatchify deleted future locations.
        velocity_vision = self.net.llm2vae(hidden[packed.vision.sequence_indexes])
        velocity_vision[patch_ids < self.spatial] = 0
        velocity_vision *= self.valid_patch_scalars[patch_ids].to(velocity_vision.dtype)
        decoded = {}
        self.net._decode_action(packed, hidden, decoded)
        velocity_action = decoded["preds_action"][0]
        velocity_action *= (1 - original.action.condition_mask[0]).to(velocity_action)
        if self.raw_action_dim is not None:
            velocity_action[:, self.raw_action_dim :] = 0
        return torch.cat((velocity_vision.flatten(), velocity_action.flatten())), patch_ids

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
        if self.config.dynamic_source == "observation":
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
        shape = clean.x0_tokens_vision[0].shape
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
        self.selector.begin(patches[self.spatial :].reshape(8, self.spatial, -1), observation=obs_patches)
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
        self.selector.complete({b: self.current_scores[b] for b in self.config.global_blocks})
        expected_removed = 8 * self.spatial - int(self.mask.sum())
        for row in self.token_rows:
            expected = 9 * self.spatial + 33
            if not (row["step"] == 0 and row["branch"] == "conditional" and row["block"] < 2):
                expected -= expected_removed
            if row["gen_tokens"] != expected:
                raise AssertionError("Real transformer input length inconsistent with selected mask")
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
            "force_full": force_full,
        }
        self.last_info = info
        self.last_sparse_latent = final_patches.detach().cpu()
        self.last_patch_ids = patch_ids.detach().cpu()
        self.last_token_rows = self.token_rows
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            if observation_rgb is not None:
                np.savez_compressed(
                    output_dir / "observation_dynamic.npz",
                    rgb=observation_rgb.cpu().numpy(),
                    dynamic_mask=self.selector.last_dynamic_mask.cpu().numpy(),
                    selected_mask=self.mask.cpu().numpy(),
                    similarities=(
                        np.full((8, self.spatial), np.nan)
                        if self.selector.last_similarities is None
                        else self.selector.last_similarities.cpu().numpy()
                    ),
                )
            grids = {
                f"conditional_step0_B{b}": scatter_future_grid(
                    score.flatten()[torch.isfinite(score.flatten())],
                    torch.where(torch.isfinite(score.flatten()))[0],
                    frames=8,
                    height=gh,
                    width=gw,
                )
                for b, score in self.current_scores.items()
            }
            selected = {name: self.mask.reshape(self.grid).cpu().numpy() for name in grids}
            export_heatmaps(grids, output_dir / "heatmaps", metadata=info, selected_masks=selected)
            torch.save(
                {
                    "patch_ids": self.last_patch_ids,
                    "patches": self.last_sparse_latent,
                    "action": action_external.cpu(),
                    "mask": self.mask.cpu(),
                    "original_grid": (9, gh, gw),
                },
                output_dir / "output.pt",
            )
            (output_dir / "token_rows.json").write_text(json.dumps(self.token_rows, indent=2))
        self.chunk += 1
        result = {"action": [action_external], "vision": []}
        if force_full:
            # Dense regression only; never called for deleted-token inference.
            dense = (
                final_patches.reshape(9, gh, gw, p, p, c)
                .permute(5, 0, 1, 3, 2, 4)
                .reshape(c, 9, gh * p, gw * p)[:, :, :h, :w]
            )
            result["vision"] = [dense]
        return result
