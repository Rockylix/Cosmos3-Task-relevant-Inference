#!/usr/bin/env python3
"""Visualize V5.1 score-smoothed, budgeted execution ROI masks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

STAGES = ("execution_masks",)


def _jaccard(left: torch.Tensor, right: torch.Tensor) -> float:
    union = int((left | right).sum())
    return float((left & right).sum() / union) if union else 1.0


def _plot_masks(masks: torch.Tensor, title: str, output: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
    for frame, axis in enumerate(axes.flat):
        image = masks[frame].reshape(17, 20).float().numpy()
        axis.imshow(image, cmap="coolwarm", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(f"L{frame + 1}: {int(masks[frame].sum())}/340")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_overlay(masks: torch.Tensor, frames_dir: Path, title: str, output: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for latent, axis in enumerate(axes.flat, start=1):
        frame_path = frames_dir / f"frame_{latent * 4:03d}.png"
        frame = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.float32) / 255.0
        mask = Image.fromarray((masks[latent - 1].reshape(17, 20).numpy() * 255).astype(np.uint8))
        mask = np.asarray(mask.resize((frame.shape[1], frame.shape[0]), Image.Resampling.NEAREST)) > 0
        overlay = frame.copy()
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.asarray([1.0, 0.0, 0.0])
        overlay[~mask] *= 0.42
        axis.imshow(overlay)
        axis.set_title(f"L{latent}: {int(masks[latent - 1].sum())}/340")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=True)
    scores = artifact["normalized_scores"].float()
    rows = []
    summary = {}
    for stage in STAGES:
        masks = artifact[stage].bool()
        stage_summary = []
        for group in range(int(masks.shape[0])):
            frame_counts = [int(value) for value in masks[group].sum(dim=-1)]
            adjacent_jaccard = [_jaccard(masks[group, frame], masks[group, frame + 1]) for frame in range(7)]
            flips = masks[group, 1:] ^ masks[group, :-1]
            flip_rate = float(flips.float().mean())
            coverages = []
            for frame in range(8):
                selected = scores[group, frame][masks[group, frame]].sum()
                coverages.append(float(selected / scores[group, frame].sum().clamp_min(1e-12)))
                rows.append(
                    {
                        "stage": stage,
                        "group": group,
                        "frame": frame + 1,
                        "tokens": frame_counts[frame],
                        "score_coverage": coverages[-1],
                    }
                )
            stage_summary.append(
                {
                    "group": group,
                    "frame_token_counts": frame_counts,
                    "mean_tokens": float(np.mean(frame_counts)),
                    "adjacent_jaccard_mean": float(np.mean(adjacent_jaccard)),
                    "adjacent_flip_rate": flip_rate,
                    "score_coverage_mean": float(np.mean(coverages)),
                }
            )
            _plot_masks(
                masks[group],
                f"{stage} | G{group + 1}",
                args.output_dir / f"{stage}_g{group + 1}.png",
            )
            if args.frames_dir is not None and stage == "execution_masks":
                _plot_overlay(
                    masks[group],
                    args.frames_dir,
                    f"V5.1 execution ROI overlay | G{group + 1}",
                    args.output_dir / f"execution_overlay_g{group + 1}.png",
                )
        summary[stage] = stage_summary
    with (args.output_dir / "mask_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "mask_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
