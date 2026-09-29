"""Functional ToCa execution boundaries. No scoring or cache-schedule changes."""

import torch

from cosmos_framework.data.generator.sequence_packing.runtime import from_und_gen_splits, get_gen_seq, get_und_seq
from cosmos_framework.model.attention import attention
from cosmos_framework.model.attention.masks import CausalType


def full_joint_dispatch(q_pack, k_pack, v_pack, normalized_keys, memory, future, scale):
    """Original joint kernel on REAL rows; padding never affects probabilities."""
    from cosmos_framework.inference.toca_joint_attention import joint_attention_score

    nq, nu = q_pack["_num_full_tokens"], q_pack["_num_causal_tokens"]
    q, kg, vg = get_gen_seq(q_pack)[:nq], get_gen_seq(k_pack)[:nq], get_gen_seq(v_pack)[:nq]
    qu = get_und_seq(q_pack)
    if memory is not None and memory.und_k_cached is not None:
        if qu.shape[0] != 0:
            raise RuntimeError("Cached UND path must be GEN-only")
        ku, vu = memory.und_k_cached[0], memory.und_v_cached[0]
        und_out = qu.new_empty(0, q.shape[1] * q.shape[2])
    else:
        ku = get_und_seq(normalized_keys if normalized_keys is not None else k_pack)[:nu]
        vu = get_und_seq(v_pack)[:nu]
        if nu:
            und_out = attention(query=qu[:nu][None], key=get_und_seq(k_pack)[:nu][None],
                                value=vu[None], is_causal=True, causal_type=CausalType.DontCare)[0].flatten(-2, -1)
        else:
            und_out = q.new_empty(0, q.shape[1] * q.shape[2])
    out, score = joint_attention_score(q, ku, kg, vu, vg, future, scale)
    # Shape restoration is outside the actual AV computation and score reduction.
    und_out = torch.cat((und_out, und_out.new_zeros(qu.shape[0] - und_out.shape[0], und_out.shape[-1])))
    gen_out = out.flatten(-2, -1)
    gen_out = torch.cat((gen_out, gen_out.new_zeros(get_gen_seq(q_pack).shape[0] - nq, gen_out.shape[-1])))
    return from_und_gen_splits(und_out, gen_out, q_pack), score


def cached_forward(layer, x, cos, sin, und_k, und_v, future, protected, indices, cached_attn, cached_mlp):
    """Same cached equations, tensor-only and out-of-place persistent cache update."""
    from cosmos_framework.inference.toca_future import apply_selected_rope
    from cosmos_framework.model.generator.mot.unified_mot import _run_mlp

    attn = layer.self_attn
    normed = layer.input_layernorm_moe_gen(x)
    q = attn.q_norm_moe_gen(attn.q_proj_moe_gen(normed.index_select(0, protected)).view(
        -1, attn.num_attention_heads, attn.head_dim))
    k = attn.k_norm_moe_gen(attn.k_proj_moe_gen(normed).view(-1, attn.num_key_value_heads, attn.head_dim))
    v = attn.v_proj_moe_gen(normed).view(-1, attn.num_key_value_heads, attn.head_dim)
    q, k = apply_selected_rope(attn, q, k, cos, sin, protected)
    out = attention(query=q.unsqueeze(0), key=torch.cat((und_k, k.unsqueeze(0)), 1),
                    value=torch.cat((und_v, v.unsqueeze(0)), 1), is_causal=False, return_lse=False)
    live_attn = attn.o_proj_moe_gen(out.squeeze(0).flatten(-2, -1))
    residual = torch.empty_like(x).index_copy(0, protected, live_attn).index_copy(0, future, cached_attn)
    z = x + residual
    selected = torch.cat((protected, future.index_select(0, indices)))
    mlp, metadata = _run_mlp(layer.mlp_moe_gen, layer.post_attention_layernorm_moe_gen(z.index_select(0, selected)))
    next_mlp = cached_mlp.index_copy(0, indices, mlp[len(protected):])
    residual = torch.empty_like(x).index_copy(0, protected, mlp[:len(protected)]).index_copy(0, future, next_mlp)
    return z + residual, next_mlp, metadata
