#!/usr/bin/env python3
"""Build compact cross-task contact sheets from per-task mask overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlays-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", action="append", nargs=2, metavar=("DIRECTORY", "TITLE"), required=True)
    parser.add_argument("--tile-width", type=int, default=1400)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for group in range(1, 4):
        tiles = []
        for directory, title in args.task:
            image = Image.open(args.overlays_root / directory / f"category_overlay_g{group}.png").convert("RGB")
            height = round(image.height * args.tile_width / image.width)
            image = image.resize((args.tile_width, height), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (args.tile_width, height + 52), "white")
            tile.paste(image, (0, 52))
            ImageDraw.Draw(tile).text((18, 16), title, fill="black")
            tiles.append(tile)

        tile_height = max(tile.height for tile in tiles)
        sheet = Image.new("RGB", (args.tile_width * 2, tile_height * 2), "white")
        for index, tile in enumerate(tiles):
            sheet.paste(tile, ((index % 2) * args.tile_width, (index // 2) * tile_height))
        sheet.save(args.output_dir / f"four_task_overlay_g{group}.png")


if __name__ == "__main__":
    main()
