#!/usr/bin/env python3
"""Plot V6-B execution masks and selected-core-block spatial attention."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image

COLORS = {
    1: np.asarray([1.0, 0.05, 0.05]),
    2: np.asarray([0.05, 0.35, 1.0]),
    3: np.asarray([1.0, 0.85, 0.05]),
    4: np.asarray([0.65, 0.65, 0.65]),
}
NAMES = {1: "core", 2: "stable", 3: "adaptive_positive", 4: "budget_fill"}


def resize_labels(labels: torch.Tensor, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(labels.reshape(17, 20).numpy().astype(np.uint8))
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST))


def plot_overlay(
    labels: torch.Tensor,
    masks: torch.Tensor,
    frames_dir: Path,
    output: Path,
    title: str,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for latent, axis in enumerate(axes.flat, start=1):
        frame = np.asarray(
            Image.open(frames_dir / f"frame_{latent * 4:03d}.png").convert("RGB"), dtype=np.float32
        ) / 255.0
        enlarged = resize_labels(labels[latent - 1], frame.shape[1], frame.shape[0])
        overlay = frame.copy()
        overlay[enlarged == 0] *= 0.38
        for value, color in COLORS.items():
            selected = enlarged == value
            overlay[selected] = 0.55 * overlay[selected] + 0.45 * color
        axis.imshow(overlay)
        axis.set_title(f"L{latent}: {int(masks[latent - 1].sum())}/340", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title + " | red=Core blue=Stable yellow=Adaptive gray=Fill")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_attention_grid(
    profiles: torch.Tensor,
    blocks: list[int],
    selected_blocks: list[int],
    output: Path,
    title: str,
) -> list[dict[str, float | int]]:
    if not selected_blocks:
        raise RuntimeError("Artifact has no selected core blocks")
    lookup = {block: index for index, block in enumerate(blocks)}
    figure, axes = plt.subplots(
        len(selected_blocks), 8, figsize=(20, 2.5 * len(selected_blocks)), constrained_layout=True, squeeze=False
    )
    selected_values = torch.stack([profiles[lookup[block]] for block in selected_blocks])
    vmax = float(torch.quantile(selected_values.reshape(-1).float(), 0.995))
    vmax = max(vmax, float(torch.finfo(selected_values.dtype).eps))
    rows: list[dict[str, float | int]] = []
    image = None
    for row_index, block in enumerate(selected_blocks):
        block_profiles = profiles[lookup[block]]
        for frame in range(8):
            values = block_profiles[frame].float()
            normalized = values / values.sum().clamp_min(torch.finfo(values.dtype).eps)
            entropy = float(
                -(normalized * normalized.clamp_min(1e-12).log()).sum() / np.log(float(values.numel()))
            )
            axis = axes[row_index, frame]
            image = axis.imshow(values.reshape(17, 20).numpy(), cmap="magma", vmin=0.0, vmax=vmax)
            axis.set_title(f"B{block} L{frame + 1}\nmass={float(values.sum()):.3g}", fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
            rows.append(
                {
                    "block": block,
                    "future_frame": frame + 1,
                    "raw_mass": float(values.sum()),
                    "raw_max": float(values.max()),
                    "normalized_spatial_entropy": entropy,
                }
            )
    if image is not None:
        figure.colorbar(image, ax=axes, fraction=0.012, pad=0.01, label="raw action-attention probability")
    figure.suptitle(title + " | shared raw scale (P99.5)")
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return rows


def _attention_overlay(
    *,
    axis: plt.Axes,
    frame_path: Path,
    values: torch.Tensor,
    vmax: float,
    title: str,
) -> None:
    frame = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.float32) / 255.0
    heat = values.reshape(17, 20).float().numpy()
    relative = np.clip(heat / max(vmax, np.finfo(np.float32).eps), 0.0, 1.0)
    alpha = 0.82 * np.sqrt(relative)
    axis.imshow(frame)
    axis.imshow(
        heat,
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        extent=(0, frame.shape[1], frame.shape[0], 0),
        alpha=alpha,
    )
    axis.set_title(title, fontsize=8)
    axis.set_xticks([])
    axis.set_yticks([])


def plot_all_block_attention_overlays(
    *,
    profiles: torch.Tensor,
    blocks: list[int],
    selected_blocks: list[int],
    block_quality: torch.Tensor,
    frames_dir: Path,
    output_dir: Path,
    title: str,
) -> list[dict[str, float | int | bool]]:
    """Plot every profiled block with fair raw and spatial-shape scales."""

    if tuple(profiles.shape) != (len(blocks), 8, 340):
        raise RuntimeError(f"Unexpected profile geometry: {tuple(profiles.shape)}")
    if tuple(block_quality.shape) != (len(blocks),):
        raise RuntimeError(f"Unexpected block-quality geometry: {tuple(block_quality.shape)}")
    selected_set = set(selected_blocks)
    raw_values = profiles.float()
    frame_mass = raw_values.sum(dim=-1, keepdim=True)
    normalized_values = raw_values / frame_mass.clamp_min(torch.finfo(raw_values.dtype).eps)
    raw_vmax = max(float(torch.quantile(raw_values.reshape(-1), 0.995)), float(torch.finfo(raw_values.dtype).eps))
    normalized_vmax = max(
        float(torch.quantile(normalized_values.reshape(-1), 0.995)),
        float(torch.finfo(normalized_values.dtype).eps),
    )
    ranked = sorted(range(len(blocks)), key=lambda index: float(block_quality[index]), reverse=True)
    rank_by_block = {blocks[index]: rank + 1 for rank, index in enumerate(ranked)}
    rows: list[dict[str, float | int | bool]] = []

    for block_index, block in enumerate(blocks):
        selected = block in selected_set
        category = "selected" if selected else "unselected"
        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        per_frame = []
        for frame in range(8):
            raw = raw_values[block_index, frame]
            normalized = normalized_values[block_index, frame]
            entropy = float(
                -(normalized * normalized.clamp_min(1e-12).log()).sum() / np.log(float(normalized.numel()))
            )
            per_frame.append((raw, normalized, entropy))
            rows.append(
                {
                    "block": block,
                    "selected": selected,
                    "quality_rank": rank_by_block[block],
                    "block_quality": float(block_quality[block_index]),
                    "future_frame": frame + 1,
                    "raw_mass": float(raw.sum()),
                    "raw_max": float(raw.max()),
                    "normalized_spatial_entropy": entropy,
                }
            )

        for mode, vmax in (("raw", raw_vmax), ("shape", normalized_vmax)):
            figure, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
            for frame, axis in enumerate(axes.flat):
                raw, normalized, entropy = per_frame[frame]
                values = raw if mode == "raw" else normalized
                _attention_overlay(
                    axis=axis,
                    frame_path=frames_dir / f"frame_{(frame + 1) * 4:03d}.png",
                    values=values,
                    vmax=vmax,
                    title=f"L{frame + 1} | mass={float(raw.sum()):.4g} | H={entropy:.3f}",
                )
            scale_name = "shared raw probability" if mode == "raw" else "per-frame normalized spatial shape"
            figure.suptitle(
                f"{title} | B{block} | {category.upper()} | rank={rank_by_block[block]} "
                f"Q={float(block_quality[block_index]):.4g} | {scale_name}"
            )
            figure.colorbar(
                ScalarMappable(norm=Normalize(vmin=0.0, vmax=vmax), cmap="magma"),
                ax=axes,
                fraction=0.012,
                pad=0.01,
            )
            figure.savefig(category_dir / f"B{block:02d}_{mode}_attention_overlay.png", dpi=150)
            plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    colors = ["#d62728" if block in selected_set else "#4c78a8" for block in blocks]
    axis.bar([f"B{block}" for block in blocks], block_quality.float().numpy(), color=colors)
    axis.set_ylabel("Q = future mass x (1 - normalized spatial entropy)")
    axis.set_title(title + " | red=selected, blue=unselected")
    axis.tick_params(axis="x", rotation=45)
    figure.savefig(output_dir / "block_quality_selection.png", dpi=180)
    plt.close(figure)

    with (output_dir / "block_attention_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected_links = [
        f"- B{block}: [raw](selected/B{block:02d}_raw_attention_overlay.png) | "
        f"[shape](selected/B{block:02d}_shape_attention_overlay.png)"
        for block in blocks
        if block in selected_set
    ]
    unselected_links = [
        f"- B{block}: [raw](unselected/B{block:02d}_raw_attention_overlay.png) | "
        f"[shape](unselected/B{block:02d}_shape_attention_overlay.png)"
        for block in blocks
        if block not in selected_set
    ]
    readme = [
        f"# {title}: selected 与 unselected block attention",
        "",
        "全部图片来自同一个 Step 0 conditional request。`raw` 图片在该任务 B4-B27/L1-L8 间共享",
        "同一概率色标，可比较真实 attention mass；`shape` 图片先对每个 future frame 的 340 个",
        "空间 token 归一化，只用于比较关注形状，不能据此判断该 block 对 future video 的总依赖。",
        "",
        "[Block quality 选择图](block_quality_selection.png)",
        "",
        "## Selected blocks",
        "",
        *selected_links,
        "",
        "## Unselected blocks",
        "",
        *unselected_links,
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    (output_dir / "scale.json").write_text(
        json.dumps(
            {
                "raw_shared_vmax_p99_5": raw_vmax,
                "shape_shared_vmax_p99_5": normalized_vmax,
                "selected_blocks": selected_blocks,
                "block_quality_ranking": [blocks[index] for index in ranked],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title-prefix", default="V6-B")
    parser.add_argument("--branch", default="conditional")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=True)
    masks = artifact["selected_execution_masks"].bool()
    labels = artifact["category_labels"][str(artifact["selected_ablation_mode"])].to(torch.int8)
    if tuple(masks.shape) != (3, 8, 340) or tuple(labels.shape) != (3, 8, 340):
        raise RuntimeError(f"Unexpected mask geometry: masks={tuple(masks.shape)}, labels={tuple(labels.shape)}")
    for group in range(3):
        plot_overlay(
            labels[group],
            masks[group],
            args.frames_dir,
            args.output_dir / f"mask_overlay_g{group + 1}.png",
            f"{args.title_prefix} G{group + 1}",
        )

    branches = [str(value) for value in artifact["branches"]]
    if args.branch not in branches:
        raise RuntimeError(f"Branch {args.branch!r} not found; available={branches}")
    raw = artifact["raw_profiles"]
    branch_profiles = raw[branches.index(args.branch)]
    blocks = [int(value) for value in artifact["blocks"]]
    selected_blocks = [int(value) for value in artifact["core_blocks"]]
    rows = plot_attention_grid(
        branch_profiles,
        blocks,
        selected_blocks,
        args.output_dir / "selected_core_blocks_future_spatial_attention.png",
        f"{args.title_prefix} {args.branch} selected Core blocks",
    )
    plot_all_block_attention_overlays(
        profiles=branch_profiles,
        blocks=blocks,
        selected_blocks=selected_blocks,
        block_quality=artifact["block_quality"],
        frames_dir=args.frames_dir,
        output_dir=args.output_dir / "all_block_attention",
        title=f"{args.title_prefix} {args.branch}",
    )
    with (args.output_dir / "selected_core_block_attention_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "selected_core_blocks.json").write_text(
        json.dumps(
            {
                "branch": args.branch,
                "selected_core_blocks": selected_blocks,
                "profile_blocks": blocks,
                "raw_attention_scale": "shared P99.5 across selected blocks and L1-L8",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
