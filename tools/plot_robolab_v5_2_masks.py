#!/usr/bin/env python3
"""Plot V5.2 Core/Stable/Adaptive/Fill masks over decoded future frames."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

COLORS = {
    1: np.asarray([1.0, 0.05, 0.05]),  # Core: red
    2: np.asarray([0.05, 0.35, 1.0]),  # Stable: blue
    3: np.asarray([1.0, 0.85, 0.05]),  # Adaptive: yellow
    4: np.asarray([0.65, 0.65, 0.65]),  # Fill/ranked: gray
}
NAMES = {1: "core", 2: "stable", 3: "adaptive_positive", 4: "budget_fill"}


def _jaccard(left: torch.Tensor, right: torch.Tensor) -> float:
    union = int((left | right).sum())
    return float((left & right).sum() / union) if union else 1.0


def _resize_labels(labels: torch.Tensor, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(labels.reshape(17, 20).numpy().astype(np.uint8))
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST))


def _plot_group(labels: torch.Tensor, masks: torch.Tensor, frames_dir: Path, title: str, output: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for latent, axis in enumerate(axes.flat, start=1):
        frame = np.asarray(Image.open(frames_dir / f"frame_{latent * 4:03d}.png").convert("RGB"), dtype=np.float32)
        frame /= 255.0
        enlarged = _resize_labels(labels[latent - 1], frame.shape[1], frame.shape[0])
        overlay = frame.copy()
        overlay[enlarged == 0] *= 0.38
        for value, color in COLORS.items():
            selected = enlarged == value
            overlay[selected] = 0.55 * overlay[selected] + 0.45 * color
        counts = ", ".join(f"{NAMES[value][0].upper()}={int((labels[latent - 1] == value).sum())}" for value in COLORS)
        axis.imshow(overlay)
        axis.set_title(f"L{latent}: {int(masks[latent - 1].sum())}/340 | {counts}", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title + " | red=core blue=stable yellow=adaptive gray=fill")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title-prefix", default="V5.2")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=True)
    mode = str(artifact["selected_ablation_mode"])
    masks = artifact["selected_execution_masks"].bool()
    labels = artifact["category_labels"][mode].to(torch.int8)
    rows = []
    for group in range(3):
        _plot_group(
            labels[group],
            masks[group],
            args.frames_dir,
            f"{args.title_prefix} execution categories | G{group + 1}",
            args.output_dir / f"category_overlay_g{group + 1}.png",
        )
        for frame in range(8):
            row = {
                "mode": mode,
                "group": group,
                "frame": frame + 1,
                "tokens": int(masks[group, frame].sum()),
                "adjacent_frame_jaccard": "" if frame == 0 else _jaccard(masks[group, frame - 1], masks[group, frame]),
            }
            row.update({name: int((labels[group, frame] == value).sum()) for value, name in NAMES.items()})
            rows.append(row)
    with (args.output_dir / "overlay_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
