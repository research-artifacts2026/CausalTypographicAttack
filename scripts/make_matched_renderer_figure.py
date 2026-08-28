#!/usr/bin/env python3
"""Create a provenance-bounded flat/scene renderer comparison figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--flat", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    images = [Image.open(path).convert("RGB") for path in (args.clean, args.flat, args.scene)]
    labels = [
        "A  CLEAN\ncorrect: B / YES",
        "B  FLAT BRIDGE\nQwen-7B: B (resists)",
        "C  SCENE-INTEGRATED BRIDGE\nQwen-7B: A (target flip)",
    ]
    fig = plt.figure(figsize=(12.0, 5.9), facecolor="white")
    grid = fig.add_gridspec(2, 3, height_ratios=[4.4, 1.15], hspace=0.05, wspace=0.035)
    for index, (image, label) in enumerate(zip(images, labels)):
        axis = fig.add_subplot(grid[0, index])
        axis.imshow(image)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(2.2)
            spine.set_edgecolor((0.15, 0.32, 0.49) if index < 2 else (0.70, 0.24, 0.12))
        axis.set_title(label, loc="left", fontsize=11.2, fontweight="bold", pad=7)
    note = fig.add_subplot(grid[1, :])
    note.axis("off")
    note.text(
        0.01, 0.78,
        "Frozen intervention: identical proposition + direction-conditioned conclusion + STATUS VERIFIED; "
        "same bbox, font geometry, question, and source image.",
        fontsize=11.2, fontweight="bold", color="#17324d", va="top",
    )
    note.text(
        0.01, 0.43,
        "All-scene grounded ASR (n=300):  Qwen-7B 59.3% flat vs 64.3% scene (p=.073);  "
        "LLaVA 39.3% vs 39.3% (p=1.000).",
        fontsize=11.0, color="#242424", va="top",
    )
    note.text(
        0.01, 0.09,
        "The displayed item is the lexicographically first frozen identifier, not selected by outcome. "
        "Scene integration is deterministic synthetic compositing, not camera capture.",
        fontsize=9.7, color="#555555", va="top",
    )
    fig.suptitle("Same adversarial text, different delivery layer", fontsize=15.5, fontweight="bold", y=0.995)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
