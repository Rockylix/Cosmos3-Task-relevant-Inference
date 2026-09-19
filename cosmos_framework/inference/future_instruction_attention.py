"""Offline/capture utilities; no model mutation, pruning, or implicit defaults.

Inputs must be the post-normalization/post-RoPE tensors actually consumed by
GEN attention (including cached or GEN-normalized UND keys). A caller must
resolve instruction positions from the tokenized prompt, not equate UND with
instruction text. This implementation prioritizes validation over latency.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class InstructionMass:
    mass: torch.Tensor  # [query]; head-mean probability mass, not re-normalized
    head_mass: torch.Tensor  # [query, query_head]
    output: torch.Tensor  # [query, query_head, head_dim], pre output projection


def _finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN/Inf")


@torch.no_grad()
def instruction_attention_mass(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    instruction_indices: torch.Tensor,
    *,
    scale: float,
    visible_mask: torch.Tensor | None = None,
    query_batch_size: int = 32,
) -> InstructionMass:
    """Recompute true all-visible-key softmax, then sum instruction columns.

    q/k/v: [tokens, heads, head_dim]. GQA uses contiguous query-head groups.
    visible_mask: optional bool [Nq, Nk], True = visible. None means full GEN
    attention, never causal or neighborhood attention. Padding must be removed
    by the caller or masked explicitly. No text-only softmax or Q/K cosine.
    """
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("Expected q/k/v [tokens, heads, dim], k/v same shape")
    nq, hq, dim = q.shape
    nk, hkv, kd = k.shape
    if min(nq, hq, dim, nk, hkv) < 1 or kd != dim or hq % hkv:
        raise ValueError("Invalid head dimensions or non-integral GQA mapping")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q/k/v must be on the same device")
    if not math.isfinite(scale) or scale <= 0 or query_batch_size < 1:
        raise ValueError("scale and query_batch_size must be positive")
    if instruction_indices.ndim != 1 or instruction_indices.dtype != torch.long:
        raise ValueError("Instruction indices must be a one-dimensional int64 tensor")
    indices = instruction_indices.to(q.device)
    if indices.numel() == 0:
        raise ValueError("No instruction tokens: do not synthesize unconditional scores")
    if int(indices.min()) < 0 or int(indices.max()) >= nk:
        raise ValueError("Instruction index out of bounds")
    if indices.unique().numel() != indices.numel():
        raise ValueError("Duplicate instruction indices would double count mass")
    if visible_mask is not None:
        if visible_mask.shape != (nq, nk) or visible_mask.dtype != torch.bool:
            raise ValueError("visible_mask must be bool [query, key]")
        visible_mask = visible_mask.to(q.device)
        if not bool(visible_mask.any(dim=-1).all()):
            raise ValueError("An attention row has no visible keys")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        _finite(name, tensor)

    # [head, key, dim], using the same contiguous GQA association as attention.
    kh = k.float().repeat_interleave(hq // hkv, dim=1).transpose(0, 1)
    vh = v.float().repeat_interleave(hq // hkv, dim=1).transpose(0, 1)
    mass_parts, output_parts = [], []
    for start in range(0, nq, query_batch_size):
        query = q[start : start + query_batch_size].float().transpose(0, 1)
        logits = torch.matmul(query, kh.transpose(-1, -2)) * scale
        if visible_mask is not None:
            logits.masked_fill_(~visible_mask[start : start + query_batch_size].unsqueeze(0), -torch.inf)
        probability = logits.softmax(dim=-1)
        mass_parts.append(probability.index_select(-1, indices).sum(-1).transpose(0, 1))
        output_parts.append(torch.matmul(probability, vh).transpose(0, 1))
    heads = torch.cat(mass_parts)
    output = torch.cat(output_parts)
    _finite("instruction mass", heads)
    _finite("recomputed attention output", output)
    if bool(((heads < -1e-6) | (heads > 1 + 1e-6)).any()):
        raise ValueError("Instruction mass outside [0,1]")
    return InstructionMass(heads.mean(dim=1), heads, output)


@torch.no_grad()
def compare_attention_output(recomputed: torch.Tensor, actual: torch.Tensor) -> dict[str, float | bool]:
    """Compare pre-O-projection kernel rows, not individual probabilities."""
    if recomputed.shape != actual.shape:
        raise ValueError("Kernel and recomputed output shapes differ")
    _finite("kernel output", actual)
    _finite("recomputed output", recomputed)
    reference = actual.float().reshape(-1)
    candidate = recomputed.float().reshape(-1)
    error = candidate - reference
    return {
        "relative_l2": float(error.norm() / reference.norm().clamp_min(1e-12)),
        "cosine": float(F.cosine_similarity(candidate, reference, dim=0, eps=1e-12)),
        "max_absolute_error": float(error.abs().max()),
        "reference_norm": float(reference.norm()),
        "zero_reference": bool(reference.norm() == 0),
    }


@torch.no_grad()
def capture_future_instruction_rows(
    *,
    q_gen: torch.Tensor,
    k_ar: torch.Tensor,
    k_gen: torch.Tensor,
    v_ar: torch.Tensor,
    v_gen: torch.Tensor,
    attn_output_gen: torch.Tensor,
    scaling: float,
    future_query_indices: torch.Tensor,
    instruction_ar_indices: torch.Tensor,
    full_gen_attention_verified: bool,
) -> tuple[InstructionMass, dict[str, float | bool]]:
    """Adapter for PackedAttentionMoT's existing attention-stats callback.

    Caller supplies resolved layout and explicitly verifies this is the ordinary
    full GEN attention path, not temporal-causal/NATTEN/extra-history attention.
    This adapter does not attach hooks, choose layers, or modify model outputs.
    """
    if not full_gen_attention_verified:
        raise ValueError("Full GEN visibility must be verified before collecting scores")
    if instruction_ar_indices.numel() == 0:
        raise ValueError("No instruction span for this branch")
    if int(instruction_ar_indices.min()) < 0 or int(instruction_ar_indices.max()) >= len(k_ar):
        raise ValueError("Instruction must index AR text, never GEN or action keys")
    if future_query_indices.ndim != 1 or future_query_indices.dtype != torch.long:
        raise ValueError("Future query indices must be an int64 vector")
    if future_query_indices.numel() == 0 or future_query_indices.unique().numel() != future_query_indices.numel():
        raise ValueError("Future query indices must be nonempty and unique")
    if int(future_query_indices.min()) < 0 or int(future_query_indices.max()) >= len(q_gen):
        raise ValueError("Future query index out of range")
    if attn_output_gen.shape != q_gen.shape:
        raise ValueError("Expected kernel output before O projection, shaped like q_gen")
    indices = future_query_indices.to(q_gen.device)
    result = instruction_attention_mass(
        q_gen.index_select(0, indices),
        torch.cat((k_ar, k_gen), dim=0),
        torch.cat((v_ar, v_gen), dim=0),
        instruction_ar_indices,
        scale=scaling,
    )
    metrics = compare_attention_output(result.output, attn_output_gen.index_select(0, indices))
    return result, metrics


def scatter_future_grid(
    mass: torch.Tensor, original_ids: torch.Tensor, *, frames: int, height: int, width: int
) -> np.ndarray:
    """Display scatter only: missing positions are NaN, never solver tokens.

    original_ids are flattened future-only (frame,y,x) IDs, excluding L0.
    This is not latent restoration and the returned grid must not enter inference.
    """
    if mass.ndim != 1 or original_ids.shape != mass.shape or original_ids.dtype != torch.long:
        raise ValueError("Expected aligned one-dimensional mass and int64 original IDs")
    if min(frames, height, width) <= 0 or mass.numel() == 0:
        raise ValueError("Empty or invalid future layout")
    _finite("observed scores", mass)
    ids = original_ids.detach().cpu()
    if int(ids.min()) < 0 or int(ids.max()) >= frames * height * width:
        raise ValueError("Original future ID out of bounds")
    if ids.unique().numel() != ids.numel():
        raise ValueError("Original future IDs must be unique")
    grid = np.full(frames * height * width, np.nan, dtype=np.float32)
    grid[ids.numpy()] = mass.detach().float().cpu().numpy()
    return grid.reshape(frames, height, width)


def export_heatmaps(
    grids: dict[str, np.ndarray],
    output_dir: Path,
    *,
    metadata: dict,
    selected_masks: dict[str, np.ndarray] | None = None,
) -> list[Path]:
    """One figure per record; every record/frame uses one shared raw color scale.

    Metadata must document task/chunk/branch/step/block for each record and the
    resolved instruction tokens. This function never derives an instruction span.
    Existing artifacts are not overwritten. NaN means unobserved, not zero mass.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not grids:
        raise ValueError("No attention grids to export")
    for name, grid in grids.items():
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("Record names must be filename-safe")
        if grid.ndim != 3 or not np.isfinite(grid).any() or np.isinf(grid).any():
            raise ValueError("Grid must be [frame,height,width] with observed finite scores")
        observed = grid[np.isfinite(grid)]
        if observed.min() < -1e-6 or observed.max() > 1 + 1e-6:
            raise ValueError("Grid is not raw attention probability mass")
        if selected_masks is not None and name in selected_masks:
            if selected_masks[name].shape != grid.shape or selected_masks[name].dtype != np.bool_:
                raise ValueError("Selected mask must be boolean and match grid")
    vmax = max(float(np.nanmax(grid)) for grid in grids.values())
    vmax = max(vmax, 1e-12)
    output_dir = Path(output_dir)
    expected = [output_dir / "scores.npz", output_dir / "metadata.json"]
    expected += [output_dir / f"{name}.png" for name in grids]
    if any(path.exists() for path in expected):
        raise FileExistsError("Refusing to overwrite heatmap artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(grids)
    if selected_masks:
        payload.update({f"selected__{name}": mask for name, mask in selected_masks.items()})
    np.savez_compressed(output_dir / "scores.npz", **payload)
    details = {
        "metadata": metadata,
        "formula": "mean_heads(sum_instruction_keys(softmax_over_all_visible_keys(QK*scale)))",
        "color_vmin": 0.0,
        "color_vmax": vmax,
        "missing_positions": "NaN in scores; gray in plots; never solver restoration",
        "statistics": {
            name: [
                {"mean": float(np.nanmean(frame)), "max": float(np.nanmax(frame))}
                if np.isfinite(frame).any()
                else {"mean": None, "max": None}
                for frame in grid
            ]
            for name, grid in grids.items()
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(details, indent=2, ensure_ascii=False, allow_nan=False))
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("#909090")
    for name, grid in grids.items():
        frames, height, width = grid.shape
        columns = min(4, frames)
        rows = math.ceil(frames / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.8 * rows), squeeze=False, layout="constrained")
        for f, ax in enumerate(axes.flat):
            if f >= frames:
                ax.set_visible(False)
                continue
            im = ax.imshow(grid[f], vmin=0, vmax=vmax, cmap=cmap, interpolation="nearest")
            stat = details["statistics"][name][f]
            label = "unobserved" if stat["mean"] is None else f"mean={stat['mean']:.4g}, max={stat['max']:.4g}"
            ax.set_title(f"L{f + 1}: {label}")
            ax.set_xlabel("spatial x")
            ax.set_ylabel("spatial y")
            if selected_masks is not None and name in selected_masks:
                for y, x in np.argwhere(selected_masks[name][f]):
                    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor="cyan", linewidth=0.35))
        fig.colorbar(im, ax=axes.ravel().tolist(), label="Raw attention mass to instruction")
        fig.suptitle(name + " | cyan: selected; gray: unobserved")
        fig.savefig(output_dir / f"{name}.png", dpi=150)
        plt.close(fig)
    return expected
