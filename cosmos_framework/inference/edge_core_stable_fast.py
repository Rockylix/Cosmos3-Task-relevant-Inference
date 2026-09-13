"""Tensor-only ASI relevance path for fixed-layout compiled decoder inference."""

from __future__ import annotations

from functools import lru_cache

import torch

from cosmos_framework.model.attention import attention


def profile_geometry(layout):
    """Validate contiguity once in Python, rather than constructing GPU indices per layer."""
    future = [p for frame in range(1, 9) for p in layout["latent_positions"][f"L{frame}"]]
    actions = [q["gen_position"] for q in layout["action_queries"] if q["query_role"] == "predicted"]
    spatial = len(layout["latent_positions"]["L1"])
    if len(future) != 8 * spatial or future != list(range(future[0], future[0] + len(future))):
        raise ValueError("Compiled ASI requires contiguous future-frame tokens")
    if len(actions) != 32 or actions != list(range(actions[0], actions[0] + 32)):
        raise ValueError("Compiled ASI requires 32 contiguous predicted-action tokens")
    return actions[0], future[0], spatial


def action_aligned_future_profiles(q_gen, k_ar, k_gen, v_ar, v_gen, scaling, geometry):
    """Same LSE-based action relevance, batched over eight frames, with no host reads.

    Nonfinite/negative profiles are checked by build_core_stable_plan once all
    28 layers finish. No per-layer item(), index uploads or Python list indexing.
    """
    action_start, future_start, spatial = geometry
    heads, width = q_gen.shape[1:]
    kv_heads = k_gen.shape[1]
    q_action = q_gen[action_start : action_start + 32]
    _, lse = attention(
        query=q_action.unsqueeze(0),
        key=torch.cat((k_ar, k_gen), dim=0).unsqueeze(0),
        value=torch.cat((v_ar, v_gen), dim=0).unsqueeze(0),
        scale=scaling,
        return_lse=True,
    )
    # Keep the reference float32 dot product and reduction order within a frame.
    queries = q_action.float().reshape(8, 4, heads, width).permute(0, 2, 1, 3).contiguous()
    keys = k_gen[future_start : future_start + 8 * spatial].float()
    keys = keys.repeat_interleave(heads // kv_heads, dim=1)
    keys = keys.reshape(8, spatial, heads, width).permute(0, 2, 3, 1).contiguous()
    logits = torch.matmul(queries * scaling, keys)
    log_normalizer = lse.reshape(8, 4, heads).permute(0, 2, 1).unsqueeze(-1).float()
    probabilities = torch.exp(logits - log_normalizer).mean(dim=1)
    weights = torch.where(torch.arange(4, device=probabilities.device) % 3 == 0, 1.0 / 6, 1.0 / 3)
    return (probabilities * weights[None, :, None]).sum(dim=1)


@lru_cache(maxsize=2)
def compiled_profile_kernel(cuda_graphs):
    return torch.compile(
        action_aligned_future_profiles,
        fullgraph=True,
        dynamic=False,
        mode="reduce-overhead" if cuda_graphs else "default",
    )
