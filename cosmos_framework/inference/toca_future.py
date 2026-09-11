"""Eager, request-local ToCa-Future adaptation for the Edge DROID policy.

Only future attention/MLP residuals may be reused across diffusion steps.
Condition/action queries always use current, complete GEN K/V plus native UND
cache. No action scores, ASI masks, or cross-chunk features are used. The default
reference backend retains native attention; the opt-in joint backend shares QK
between attention output and scoring. Only an explicit caller installs this
controller.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial

import torch

from cosmos_framework.data.generator.sequence_packing.runtime import from_und_gen_splits, get_gen_seq, get_und_seq
from cosmos_framework.inference.edge_core_stable_layout import _action_query_layout
from cosmos_framework.model.attention import attention


@dataclass(frozen=True)
class ToCaFutureConfig:
    full_steps: tuple[int, ...] = (0, 2)
    fresh_ratio: float = 0.25
    layer_slope: float = 0.5
    age_weight: float = 0.25
    period: int = 2
    score_query_tile: int = 256
    attention_backend: str = "reference"
    spatial_bonus: float = 0.0
    cfg_selection: str = "shared"

    def __post_init__(self):
        if not self.full_steps or self.full_steps[0] != 0:
            raise ValueError("The first denoise step must be full")
        if tuple(sorted(set(self.full_steps))) != self.full_steps or any(s not in range(4) for s in self.full_steps):
            raise ValueError("full_steps must be sorted unique indices in [0,3]")
        if not 0 <= self.fresh_ratio <= 1 or not 0 <= self.layer_slope <= 1:
            raise ValueError("Invalid fresh ratio or layer slope")
        if self.period < 1 or self.score_query_tile < 1 or self.age_weight < 0:
            raise ValueError("Invalid cache period, score tile, or age weight")
        if self.attention_backend not in ("reference", "joint"):
            raise ValueError("attention_backend must be reference or joint")
        if not 0 <= self.spatial_bonus <= 1 or self.cfg_selection not in ("shared", "independent"):
            raise ValueError("Invalid spatial bonus or CFG selection")

    def fresh_count(self, block: int, blocks: int, future_count: int) -> int:
        rho = self.fresh_ratio * (1 + self.layer_slope - 2 * self.layer_slope * block / max(blocks - 1, 1))
        return int(max(0.0, min(1.0, rho)) * future_count)


def spatial_windows(frames: int, height: int, width: int, device):
    """Nonoverlapping frame-local 2x2 windows; -1 marks invalid edge cells."""
    if min(frames, height, width) < 1:
        raise ValueError("Invalid future grid")
    ids = torch.arange(frames * height * width, device=device).reshape(frames, height, width)
    ids = torch.nn.functional.pad(ids, (0, width % 2, 0, height % 2), value=-1)
    return ids.reshape(frames, (height + 1) // 2, 2, (width + 1) // 2, 2).permute(0, 1, 3, 2, 4).reshape(-1, 4)


def select_fresh(scores: torch.Tensor, age: torch.Tensor, count: int, config: ToCaFutureConfig, windows=None):
    """Normalize, add age, optionally boost each spatial-window winner, rank."""
    if scores.ndim != 1 or scores.shape != age.shape or not 0 <= count <= scores.numel():
        raise ValueError("Invalid score/age layout or refresh count")
    importance = torch.nn.functional.normalize(scores.float(), dim=0) + config.age_weight * age / config.period
    if config.spatial_bonus:
        if windows is None:
            raise ValueError("Spatial bonus requires real frame-local grid indices")
        values = importance[windows.clamp_min(0)].masked_fill(windows < 0, -torch.inf)
        winner = windows.gather(1, values.argmax(1, keepdim=True)).flatten()
        importance = importance.scatter(0, winner, importance[winner] * (1 + config.spatial_bonus))
    # Stable ordering makes exact ties deterministic without random perturbations.
    indices = importance.argsort(descending=True, stable=True)[:count]
    next_age = age + 1
    next_age.index_fill_(0, indices, 0)
    return indices, next_age


def incoming_future_score(q, k_und, k_gen, future_positions, scale: float, tile: int = 256):
    """Exact all-GEN-query key-column statistic, bounded by a query tile.

    The denominator includes ALL real UND and GEN keys. Accumulation and
    softmax are FP32. Native attention remains responsible for model outputs;
    this additional QK/softmax reduction is intentionally charged to ToCa.
    """
    if q.ndim != 3 or k_gen.ndim != 3 or k_und.ndim != 3:
        raise ValueError("Expected [tokens,heads,head_dim] Q/K")
    if q.shape[0] != k_gen.shape[0] or q.shape[1] % k_gen.shape[1] or tile < 1:
        raise ValueError("Unsupported Q/K layout or query tile")
    keys = torch.cat((k_und, k_gen), dim=0)
    keys = keys.repeat_interleave(q.shape[1] // keys.shape[1], dim=1).transpose(0, 1).float()
    columns = future_positions + k_und.shape[0]
    total = torch.zeros(len(columns), device=q.device, dtype=torch.float32)
    for start in range(0, q.shape[0], tile):
        logits = torch.matmul(q[start : start + tile].transpose(0, 1).float(), keys.transpose(-1, -2)) * scale
        probabilities = logits.softmax(dim=-1)
        total += probabilities.index_select(-1, columns).sum(dim=(0, 1))
    return total / (q.shape[0] * q.shape[1])


def apply_selected_rope(module, query, key, cos, sin, protected):
    """Use the model's original RoPE function and unrenumbered positions."""
    q_cos, q_sin = cos.index_select(0, protected), sin.index_select(0, protected)
    query, _ = module._apply_rotary_pos_emb(query, query[:, :0], q_cos, q_sin, unsqueeze_dim=1)
    _, key = module._apply_rotary_pos_emb(key[:, :0], key, cos, sin, unsqueeze_dim=1)
    return query, key


class ToCaFutureController:
    """Temporarily route decoder forwards, leaving the disabled path untouched."""

    def __init__(self, net, config: ToCaFutureConfig | None = None, *, num_steps=4, guidance=3.0):
        if num_steps != 4 or guidance != 3.0:
            raise ValueError("This experiment requires four steps and CFG=3")
        self.net = net
        self.config = config or ToCaFutureConfig()
        self.model = net.language_model.model
        self.layers = list(self.model.layers)
        if len(self.layers) != 28:
            raise ValueError("This controller is validated only for Edge's 28 blocks")
        self.current = None
        self.calls = 0
        self.layout = None
        self.future = None
        self.protected = None
        self.windows = None
        self.cache = {}
        self.scores = {}
        self.ages = {}
        self.indices = {}
        self.records = []
        self.finite = []
        self._handles = []
        self._originals = []
        self._block_sequence = []

    def __enter__(self):
        if getattr(self.model, "_robolab_version1_controller", None) is not None:
            raise RuntimeError("ToCa-Future and ASI are mutually exclusive")
        if getattr(self.model, "_toca_future_controller", None) is not None:
            raise RuntimeError("Nested ToCa controllers are unsupported")
        for layer in self.layers:
            attn = layer.self_attn
            if any(
                getattr(attn, key, None) is not None
                for key in (
                    "_attention_stats_capture_callback",
                    "_sparse_qk_projection_callback",
                    "_rope_qk_capture_callback",
                )
            ):
                raise RuntimeError("ToCa conflicts with existing attention instrumentation")
        self.model._toca_future_controller = self
        self._handles.append(self.net.register_forward_pre_hook(self._pre, with_kwargs=True))
        self._handles.append(self.net.register_forward_hook(self._post, with_kwargs=True, always_call=True))
        for block, layer in enumerate(self.layers):
            original = layer.forward
            self._originals.append((layer, "forward" in layer.__dict__, original))
            layer.forward = partial(self._forward, block, original)
        return self

    def __exit__(self, *exc):
        for layer, had_override, original in reversed(self._originals):
            if had_override:
                layer.forward = original
            else:
                del layer.forward
        self._originals.clear()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if getattr(self.model, "_toca_future_controller", None) is self:
            del self.model._toca_future_controller
        self.current = None

    def _pre(self, module, args, kwargs):
        del module
        if kwargs.get("und_only", args[2] if len(args) > 2 else False):
            self.current = None
            return
        if self.calls >= 8:
            raise RuntimeError("More than eight denoising forwards")
        packed = args[0] if args else kwargs["packed_seq"]
        layout = _action_query_layout(torch, packed)
        if self.layout is None:
            self.layout = layout
        elif layout != self.layout:
            raise RuntimeError("Token layout changed inside a request")
        self.current = (self.calls // 2, "conditional" if self.calls % 2 == 0 else "unconditional")
        self.calls += 1
        self._block_sequence = []

    def _post(self, module, args, kwargs, output):
        del module, args, kwargs
        if output is not None and self.current is not None and self._block_sequence != list(range(28)):
            raise RuntimeError("Incomplete or reordered ToCa Transformer stack")
        self.current = None

    def _init_positions(self, x):
        if self.future is not None:
            return
        assert self.layout is not None
        positions = [p for f in range(1, 9) for p in self.layout["latent_positions"][f"L{f}"]]
        self.future = torch.tensor(positions, device=x.device, dtype=torch.long)
        keep = torch.ones(self.layout["num_gen_tokens"], device=x.device, dtype=torch.bool)
        keep[self.future] = False
        self.protected = keep.nonzero().flatten()
        protected_expected = self.layout["latent_positions"]["L0"] + self.layout["action_positions"]
        if sorted(protected_expected) != self.protected.tolist():
            raise RuntimeError("L0/action protection does not partition GEN")
        if self.config.spatial_bonus:
            frames, height, width = self.layout["latent_shape_thw"]
            if frames != 9 or len(self.future) != 8 * height * width:
                raise RuntimeError("Spatial bonus requires the validated eight-future-frame layout")
            self.windows = spatial_windows(8, height, width, x.device)

    def _forward(
        self,
        block,
        original,
        input,
        attention_mask,
        packed_position_embeddings,
        natten_metadata=None,
        memory_value=None,
        gen_only=False,
    ):
        if self.current is None:
            return original(input, attention_mask, packed_position_embeddings, natten_metadata, memory_value, gen_only)
        if natten_metadata is not None or input.get("is_sharded", False):
            raise RuntimeError("ToCa smoke supports single-GPU full attention only")
        x = get_gen_seq(input)
        if len(x) != input["_num_full_tokens"]:
            raise RuntimeError("ToCa eager reference does not support padded GEN rows")
        self._init_positions(x)
        step, branch = self.current
        if step in self.config.full_steps:
            result = self._full(
                block, original, input, attention_mask, packed_position_embeddings, memory_value, gen_only
            )
            fresh = len(self.future)
            kind = "full"
        else:
            result, fresh = self._cached(block, input, packed_position_embeddings, memory_value, gen_only)
            kind = "cached"
        self._block_sequence.append(block)
        self.finite.append(torch.isfinite(get_gen_seq(result[0])).all())
        self.records.append(
            {
                "step": step,
                "branch": branch,
                "block": block,
                "mode": kind,
                "gen_tokens": len(x),
                "future_tokens": len(self.future),
                "protected_tokens": len(self.protected),
                "q_rows": len(x) if kind == "full" else len(self.protected),
                "kv_rows": len(x),
                "o_rows": len(x) if kind == "full" else len(self.protected),
                "mlp_rows": len(self.protected) + fresh,
                "future_mlp_fresh": fresh,
            }
        )
        return result

    def _full(self, block, original, input, mask, rope, memory, gen_only):
        layer = self.layers[block]
        _, branch = self.current
        entry = {}
        self.cache[branch, block] = entry

        def capture_attn(module, args, output):
            del module, args
            entry["attn"] = get_gen_seq(output[0]).index_select(0, self.future).detach().clone()

        def capture_mlp(module, args, output):
            del module, args
            value = output[0] if isinstance(output, tuple) else output
            entry["mlp"] = value.index_select(0, self.future).detach().clone()

        def capture_score(**values):
            self.scores[branch, block] = incoming_future_score(
                values["q_gen"],
                values["k_ar"],
                values["k_gen"],
                self.future,
                float(values["scaling"]),
                self.config.score_query_tile,
            ).detach()

        handles = [
            layer.self_attn.register_forward_hook(capture_attn),
            layer.mlp_moe_gen.register_forward_hook(capture_mlp),
        ]
        previous_dispatch = layer.self_attn.dispatch_attention_fn
        if self.config.attention_backend == "joint":
            layer.self_attn.dispatch_attention_fn = partial(self._joint_dispatch, block)
        else:
            layer.self_attn._attention_stats_capture_callback = capture_score
        try:
            output = original(input, mask, rope, memory_value=memory, gen_only=gen_only)
        finally:
            layer.self_attn._attention_stats_capture_callback = None
            layer.self_attn.dispatch_attention_fn = previous_dispatch
            for handle in handles:
                handle.remove()
        if set(entry) != {"attn", "mlp"}:
            raise RuntimeError("Incomplete full-step cache")
        if self.config.cfg_selection == "independent" or branch == "unconditional":
            key = (branch, block) if self.config.cfg_selection == "independent" else block
            self.ages[key] = torch.zeros(len(self.future), device=self.future.device, dtype=torch.float32)
        return output

    def _select_indices(self, block):
        step, branch = self.current
        independent = self.config.cfg_selection == "independent"
        key = (step, branch, block) if independent else (step, block)
        age_key = (branch, block) if independent else block
        if independent or branch == "conditional":
            score = (
                self.scores[branch, block]
                if independent
                else (self.scores["conditional", block] + self.scores["unconditional", block]) * 0.5
            )
            count = self.config.fresh_count(block, len(self.layers), len(self.future))
            indices, age = select_fresh(score, self.ages[age_key], count, self.config, self.windows)
            self.indices[key] = indices
            self.ages[age_key] = age
        return self.indices[key]

    def _joint_dispatch(
        self,
        block,
        q_pack,
        k_pack,
        v_pack,
        mask,
        natten_metadata=None,
        memory_value=None,
        packed_key_states_normalized=None,
    ):
        """Only GEN uses joint AV/score; UND retains native causal attention."""
        from cosmos_framework.inference.toca_joint_attention import joint_attention_score
        from cosmos_framework.model.attention.masks import CausalType

        if (
            natten_metadata is not None
            or getattr(mask, "is_three_way", False)
            or getattr(mask, "control_stream_token_ranges", None) is not None
            or q_pack["sample_offsets"].numel() != 2
        ):
            raise RuntimeError("Joint ToCa supports only one sample with full GEN attention")
        q, kg, vg = get_gen_seq(q_pack), get_gen_seq(k_pack), get_gen_seq(v_pack)
        qu = get_und_seq(q_pack)
        if memory_value is not None and memory_value.und_k_cached is not None:
            if len(qu):
                raise RuntimeError("Cached UND path must be GEN-only")
            ku, vu = memory_value.und_k_cached[0], memory_value.und_v_cached[0]
            und_out = q.new_empty(0, q.shape[1] * q.shape[2])
        else:
            gen_keys = packed_key_states_normalized if packed_key_states_normalized is not None else k_pack
            ku, vu = get_und_seq(gen_keys), get_und_seq(v_pack)
            if len(qu):
                und_out = attention(
                    query=qu[None],
                    key=get_und_seq(k_pack)[None],
                    value=vu[None],
                    is_causal=True,
                    causal_type=CausalType.DontCare,
                )[0].flatten(-2, -1)
            else:
                und_out = q.new_empty(0, q.shape[1] * q.shape[2])
        out, score = joint_attention_score(q, ku, kg, vu, vg, self.future, self.layers[block].self_attn.scaling)
        self.scores[self.current[1], block] = score.detach()
        return from_und_gen_splits(und_out, out.flatten(-2, -1), q_pack), None

    def _cached(self, block, pack, rope, memory, gen_only):
        from cosmos_framework.model.generator.mot.unified_mot import _run_mlp

        if not gen_only or memory is None or memory.und_k_cached is None:
            raise RuntimeError("Cached steps require the native, populated request-local UND cache")
        if getattr(memory, "for_cuda_graphs", False):
            raise RuntimeError("CUDA graphs are disabled for this reference experiment")
        step, branch = self.current
        entry = self.cache[branch, block]
        layer = self.layers[block]
        attn = layer.self_attn
        x = get_gen_seq(pack)
        normed = layer.input_layernorm_moe_gen(x)
        q = attn.q_norm_moe_gen(
            attn.q_proj_moe_gen(normed.index_select(0, self.protected)).view(
                -1, attn.num_attention_heads, attn.head_dim
            )
        )
        k = attn.k_norm_moe_gen(attn.k_proj_moe_gen(normed).view(-1, attn.num_key_value_heads, attn.head_dim))
        v = attn.v_proj_moe_gen(normed).view(-1, attn.num_key_value_heads, attn.head_dim)
        q, k = apply_selected_rope(attn, q, k, get_gen_seq(rope[0]), get_gen_seq(rope[1]), self.protected)
        keys = torch.cat((memory.und_k_cached, k.unsqueeze(0)), dim=1)
        values = torch.cat((memory.und_v_cached, v.unsqueeze(0)), dim=1)
        out = attention(query=q.unsqueeze(0), key=keys, value=values, is_causal=False, return_lse=False)
        live_attn = attn.o_proj_moe_gen(out.squeeze(0).flatten(-2, -1))
        attn_residual = torch.empty_like(x)
        attn_residual.index_copy_(0, self.protected, live_attn)
        attn_residual.index_copy_(0, self.future, entry["attn"])
        z = x + attn_residual
        indices = self._select_indices(block)
        selected = torch.cat((self.protected, self.future.index_select(0, indices)))
        mlp_out, metadata = _run_mlp(
            layer.mlp_moe_gen, layer.post_attention_layernorm_moe_gen(z.index_select(0, selected))
        )
        entry["mlp"].index_copy_(0, indices, mlp_out[len(self.protected) :])
        residual = torch.empty_like(x)
        residual.index_copy_(0, self.protected, mlp_out[: len(self.protected)])
        residual.index_copy_(0, self.future, entry["mlp"])
        output = z + residual
        packed = from_und_gen_splits(output.new_empty(0, output.shape[-1]), output, pack)
        return (packed, {"gen": metadata} if metadata is not None else {}, None), len(indices)

    def finish(self):
        if self.calls != 8 or len(self.records) != 224:
            raise RuntimeError(f"Expected 8 forwards / 224 blocks, got {self.calls} / {len(self.records)}")
        if not bool(torch.stack(self.finite).all().item()):
            raise FloatingPointError("NaN/Inf in ToCa Transformer output")
        reference = sum(r["gen_tokens"] for r in self.records)
        summary = {
            "strategy": "toca-future",
            "config": asdict(self.config),
            "forwards": self.calls,
            "block_calls": len(self.records),
            "all_intermediate_finite": True,
            "layout": self.layout,
        }
        for module, field in (("q", "q_rows"), ("kv", "kv_rows"), ("o", "o_rows"), ("mlp", "mlp_rows")):
            actual = sum(r[field] for r in self.records)
            summary[module] = {
                "computed_rows": actual,
                "dense_rows": reference,
                "saved_rows": reference - actual,
                "saved_fraction": 1 - actual / reference,
            }
        return summary
