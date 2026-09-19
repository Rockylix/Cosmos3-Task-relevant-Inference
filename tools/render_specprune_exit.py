"""Offline plots of observed-image scoring and the executed spatial plans."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    index = [
        "# 观测评分与未来帧共享mask",
        "",
        "底图为真实当前观测，不是预测future RGB。每个chunk的计划在所有step和CFG分支按层重放。",
        "",
    ]
    for file in sorted((args.run / "specprune/artifacts").glob("*/chunk_003/observation_scores.npz")):
        data = np.load(file)
        meta = json.loads((file.parent / "metadata.json").read_text())
        geometry = meta["observation_geometry"]
        gh, gw = geometry["grid_hw"]
        oh, ow = geometry["original_hw"]
        side = geometry["nominal_rgb_patch_side"]
        image = data["rgb"][:, :oh, :ow].transpose(1, 2, 0)
        masks = sorted(int(k.split("B")[1]) for k in data.files if k.startswith("mask_B"))
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
        for ax, b in zip(axes.flat, masks, strict=True):
            mask = data[f"mask_B{b}"].reshape(gh, gw)
            alpha = np.repeat(np.repeat(mask.astype(float), side, axis=0), side, axis=1)
            colors = np.zeros((*alpha.shape, 4))
            colors[:, :, :3] = [0, 1, 1]
            colors[:, :, 3] = alpha * 0.5
            ax.imshow(image)
            ax.imshow(colors)
            ax.set_xlim(-0.5, ow - 0.5)
            ax.set_ylim(oh - 0.5, -0.5)
            ax.set_title(f"After B{b}: {int(mask.sum())}/340 positions per future")
            ax.set_axis_off()
        fig.suptitle(f"{file.parent.parent.name} chunk3 | shared L1-L8 mask, not L0 pruning")
        fig.savefig(file.parent / "mask_overlay.png", dpi=140)
        plt.close(fig)
        names = [f"L0_instruction_B{b}" for b in (0, 1, 13, 27)]
        vmax = max(float(data[k].max()) for k in names)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), layout="constrained")
        for ax, name in zip(axes, names, strict=True):
            artist = ax.imshow(data[name].reshape(gh, gw), vmin=0, vmax=vmax, cmap="magma")
            ax.set_title(name)
            ax.set_xlabel("spatial x")
            ax.set_ylabel("spatial y")
        fig.colorbar(artist, ax=list(axes), label="Head-mean instruction probability mass")
        fig.savefig(file.parent / "L0_instruction_heatmaps.png", dpi=140)
        plt.close(fig)
        names = [f"action_L0_B{b}" for b in (14, 19, 24)]
        vmax = max(float(data[k].mean(0).max()) for k in names)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout="constrained")
        for ax, name in zip(axes, names, strict=True):
            artist = ax.imshow(data[name].mean(0).reshape(gh, gw), vmin=0, vmax=vmax, cmap="magma")
            ax.set_title(name + " (q1-q32 mean)")
            ax.set_xlabel("spatial x")
            ax.set_ylabel("spatial y")
        fig.colorbar(artist, ax=list(axes), label="Action-to-L0 probability per token")
        fig.savefig(file.parent / "action_L0_heatmaps.png", dpi=140)
        plt.close(fig)
        rel = file.parent.relative_to(args.run)
        index += [
            f"## {file.parent.parent.name}",
            "",
            f"- [逐层共享 mask]({rel}/mask_overlay.png)",
            f"- [L0→指令热力图]({rel}/L0_instruction_heatmaps.png)",
            f"- [action→L0热力图]({rel}/action_L0_heatmaps.png)",
            "",
        ]
    (args.run / "heatmap_index.md").write_text("\n".join(index) + "\n")


if __name__ == "__main__":
    main()
