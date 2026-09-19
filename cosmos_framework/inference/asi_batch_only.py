"""B1: batch eight frame scoring operations; retain all legacy setup/checks.

No layout/index cache, metadata caching, decoder metadata return, compilation,
Graph, changed LSE backend, deferred validation, or new pruning policy.
"""

from cosmos_framework.scripts import robolab_version1 as legacy


def batch_only_profiles(*, torch, q_gen, k_ar, k_gen, v_ar, v_gen, scaling, token_layout):
    if k_ar.shape != v_ar.shape or k_gen.shape != v_gen.shape:
        raise RuntimeError("K/V geometry differs during Version1 profiling")
    action_index, weights = legacy._validate_profile_inputs(
        torch=torch, q_gen=q_gen, k_ar=k_ar, k_gen=k_gen, token_layout=token_layout
    )
    q_heads, kv_heads = int(q_gen.shape[1]), int(k_gen.shape[1])
    q_action = q_gen.index_select(0, action_index)
    k_all, v_all = torch.cat((k_ar, k_gen), dim=0), torch.cat((v_ar, v_gen), dim=0)
    output, lse = legacy.attention(
        query=q_action.unsqueeze(0),
        key=k_all.unsqueeze(0),
        value=v_all.unsqueeze(0),
        scale=float(scaling),
        return_lse=True,
    )
    del output
    lse = lse.squeeze(0)
    if lse.ndim == 3 and int(lse.shape[-1]) == 1:
        lse = lse.squeeze(-1)
    if tuple(lse.shape) != (32, q_heads):
        raise RuntimeError(f"Unexpected attention LSE shape {tuple(lse.shape)}")
    lse_hq = lse.detach().float().transpose(0, 1)
    q_action_hqd = q_action.detach().float().permute(1, 0, 2).contiguous()
    queries, keys, normalizers = [], [], []
    # Deliberately preserve legacy per-layer/per-frame index creation and gather.
    # Only downstream eight frame math is batched, not the later B2 geometry cache.
    for latent in range(1, 9):
        horizons = list(range(4 * (latent - 1), 4 * latent))
        positions = torch.tensor(token_layout["latent_positions"][f"L{latent}"], dtype=torch.long, device=q_gen.device)
        k_frame = k_gen.index_select(0, positions).detach().float()
        k_frame = k_frame.repeat_interleave(q_heads // kv_heads, dim=1).permute(1, 0, 2).contiguous()
        queries.append(q_action_hqd[:, horizons])
        keys.append(k_frame)
        normalizers.append(lse_hq[:, horizons, None])
    logits = torch.einsum("fhqd,fhkd->fhqk", torch.stack(queries) * float(scaling), torch.stack(keys))
    probabilities = torch.exp(logits - torch.stack(normalizers)).mean(dim=1)
    result = (probabilities * weights[None, :, None]).sum(dim=1)
    # Same per-layer host checks as legacy: not folded into this iteration.
    if not bool(torch.isfinite(result).all()) or bool((result < 0).any()):
        raise RuntimeError("Version1 fused Action-Relevance profile is invalid")
    return result


class BatchOnlyController(legacy.Version1Controller):
    def _capture_profile(self, **kwargs):
        block = int(kwargs["layer_index"])
        if self._profile_callback_block != block or self._layout is None:
            raise RuntimeError("Version1 attention callback has stale block state")
        profile = batch_only_profiles(
            torch=self.torch,
            **{key: kwargs[key] for key in ("q_gen", "k_ar", "k_gen", "v_ar", "v_gen")},
            scaling=float(kwargs["scaling"]),
            token_layout=self._layout,
        ).detach()
        self._profile_records.append({"block": block, "profiles": profile})
