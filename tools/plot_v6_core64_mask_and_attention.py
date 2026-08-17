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
