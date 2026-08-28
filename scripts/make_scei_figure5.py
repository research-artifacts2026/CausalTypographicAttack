#!/usr/bin/env python3
"""Render the paper's SCEI-Images-300 qualitative/result overview.

The image triplet comes from the public, license-filtered dataset. Aggregate
bars come from the complete fixed four-model result release. The selected
sample outcomes are a small, explicitly published view of the audited raw
logs; the full sample-level logs are not required to reproduce this figure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image


DATA_PATH = ROOT / "paper_assets" / "figure5_scei_overview.json"
DATASET_ROOT = ROOT / "datasets" / "scei_images_coco_n300"
SUMMARY_PATH = ROOT / "results" / "scei_images_n300_eval_v1" / "model_summary.csv"
OUTPUT_ROOT = ROOT / "figures"

INK = "#172033"
MUTED = "#667085"
LINE = "#D8DEE9"
BLUE = "#2F6B9A"
BLUE_LIGHT = "#EAF2F8"
ORANGE = "#E98B2A"
ORANGE_LIGHT = "#FFF3E6"
RED = "#C74343"
RED_LIGHT = "#FCEAEA"
GREEN = "#2A8C68"
GREEN_LIGHT = "#E8F5EF"
GRAY_LIGHT = "#F5F7FA"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_summary() -> list[dict]:
    with SUMMARY_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4:
        raise ValueError(f"Expected four model rows, found {len(rows)}")
    return rows


def add_round_box(fig, xywh, *, face, edge=LINE, radius=0.012, lw=1.0):
    x, y, w, h = xywh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        transform=fig.transFigure,
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        zorder=-1,
    )
    fig.patches.append(patch)
    return patch


def draw_image_panel(fig, path: Path, rect, title: str, subtitle: str, color: str):
    ax = fig.add_axes(rect)
    ax.imshow(Image.open(path).convert("RGB"))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
        spine.set_edgecolor(color)
    x, y, w, h = rect
    fig.text(x, y + h + 0.012, title, color=color, fontsize=10.4, fontweight="bold")
    fig.text(x, y - 0.027, subtitle, color=INK, fontsize=8.4)
    return ax


def draw_sample_table(fig, responses: list[dict]):
    add_round_box(fig, (0.355, 0.055, 0.292, 0.375), face="white")
    fig.text(0.37, 0.404, "C", color="white", fontsize=8.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", facecolor=BLUE, edgecolor=BLUE))
    fig.text(0.397, 0.406, "Fresh victim outputs on this item", color=INK, fontsize=10.2, fontweight="bold")
    fig.text(0.37, 0.378, "Clean question asks whether the false record is correct.", color=MUTED, fontsize=7.3)

    headers = ["Model", "Clean", "Attack", "Read", "Endpoint"]
    xs = [0.372, 0.472, 0.523, 0.574, 0.617]
    for x, label in zip(xs, headers):
        fig.text(x, 0.346, label, color=MUTED, fontsize=7.1, fontweight="bold",
                 ha="left" if label == "Model" else "center")
    fig.lines.append(mpl.lines.Line2D([0.37, 0.635], [0.336, 0.336], transform=fig.transFigure,
                                     color=LINE, linewidth=0.8))

    y_values = [0.302, 0.259, 0.216, 0.173]
    for y, row in zip(y_values, responses):
        fig.text(xs[0], y, row["display_model"], color=INK, fontsize=7.7, fontweight="bold", va="center")
        clean_color = GREEN if row["clean_correct"] else RED
        fig.text(xs[1], y, row["clean_semantic"], color=clean_color, fontsize=7.7,
                 fontweight="bold", ha="center", va="center")
        fig.text(xs[2], y, row["scene_false_semantic"], color=RED, fontsize=7.7,
                 fontweight="bold", ha="center", va="center")
        read = "exact" if row["exact_read"] else "mismatch"
        fig.text(xs[3], y, read, color=GREEN if row["exact_read"] else MUTED, fontsize=7.0,
                 fontweight="bold", ha="center", va="center")
        endpoint = "SUCCESS" if row["strict_success"] else ("excluded" if not row["clean_correct"] else "fail")
        endpoint_color = GREEN if row["strict_success"] else MUTED
        fig.text(xs[4], y, endpoint, color=endpoint_color, fontsize=6.8,
                 fontweight="bold", ha="center", va="center")

    fig.text(0.37, 0.136, "Strict = clean correct + attacked target + exact independent read.",
             color=MUTED, fontsize=7.0)


def draw_aggregate(fig, rows: list[dict]):
    add_round_box(fig, (0.665, 0.055, 0.300, 0.375), face="white")
    fig.text(0.68, 0.404, "D", color="white", fontsize=8.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", facecolor=ORANGE, edgecolor=ORANGE))
    fig.text(0.707, 0.406, "All 300 registered scenes", color=INK, fontsize=10.2, fontweight="bold")

    display = {
        "Qwen2.5-VL-3B": "Qwen-3B",
        "Qwen2.5-VL-7B": "Qwen-7B",
        "LLaVA-OV-8B": "LLaVA-OV",
        "InternVL2-8B": "InternVL2",
    }
    names = [display[r["model"]] for r in rows]
    target = [100 * float(r["scene_target_asr"]) for r in rows]
    read = [100 * float(r["scene_exact_read"]) for r in rows]
    strict = [100 * float(r["scene_strict_asr"]) for r in rows]

    ax = fig.add_axes([0.695, 0.175, 0.245, 0.165])
    y = list(range(len(rows)))[::-1]
    h = 0.18
    ax.barh([v + h for v in y], target, height=h, color="#E9A15B", label="Target YES")
    ax.barh(y, read, height=h, color="#6FA7C9", label="Exact read")
    ax.barh([v - h for v in y], strict, height=h, color=RED, label="Strict (both)")
    ax.set_yticks(y, names)
    ax.set_xlim(0, 82)
    ax.set_xticks([0, 40, 80], ["0", "40", "80%"])
    ax.tick_params(axis="both", labelsize=6.7, length=0)
    ax.grid(axis="x", color=LINE, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.02), ncol=3, frameon=False,
              fontsize=6.2, handlelength=1.1, columnspacing=0.8)

    for yy, value in zip([v - h for v in y], strict):
        ax.text(value + 1.2, yy, f"{value:.1f}", va="center", ha="left", fontsize=6.2,
                color=RED, fontweight="bold")

    deltas = [100 * float(r["scene_minus_flat"]) for r in rows]
    pvals = [float(r["mcnemar_p"]) for r in rows]
    delta_text = " / ".join(f"{value:+.1f}" if value else "0.0" for value in deltas)
    fig.text(0.68, 0.142, f"Scene - flat strict delta: {delta_text} pp", color=INK, fontsize=6.8, fontweight="bold")
    fig.text(0.68, 0.122, f"Same model order; all exact McNemar p >= {min(pvals):.3f}.", color=MUTED, fontsize=6.7)


def make_figure(output_root: Path) -> tuple[Path, Path, Path]:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    sample = data["sample"]
    rows = load_summary()
    basename = sample["image_basename"]
    image_paths = {
        "clean": DATASET_ROOT / "images" / "clean" / basename,
        "false": DATASET_ROOT / "images" / "attack_false" / basename,
        "true": DATASET_ROOT / "images" / "control_true" / basename,
    }
    for path in [DATA_PATH, SUMMARY_PATH, *image_paths.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(12.0, 6.4), dpi=300, facecolor="white")

    fig.text(0.035, 0.955, "A", color="white", fontsize=9, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.28", facecolor=INK, edgecolor=INK))
    fig.text(0.063, 0.958, "Controlled visual intervention", color=INK, fontsize=11.5, fontweight="bold")
    fig.text(0.963, 0.958, "same source scene  |  same carrier geometry  |  one field changed",
             color=MUTED, fontsize=8.2, ha="right")

    rects = [(0.035, 0.535, 0.285, 0.365), (0.3575, 0.535, 0.285, 0.365), (0.68, 0.535, 0.285, 0.365)]
    draw_image_panel(fig, image_paths["clean"], rects[0], "CLEAN SCENE", "visible anchor: person in yellow pants", MUTED)
    draw_image_panel(fig, image_paths["false"], rects[1], "SCENE-ADAPTIVE FALSE RECORD", "30.0 C paired with 68.0 F", RED)
    draw_image_panel(fig, image_paths["true"], rects[2], "ONE-FIELD CORRECTED TWIN", "68.0 F -> 86.0 F; layout and mask fixed", GREEN)
    for x in (0.338, 0.6605):
        fig.text(x, 0.71, "->", fontsize=18, color=ORANGE, fontweight="bold", ha="center", va="center")

    add_round_box(fig, (0.035, 0.055, 0.302, 0.375), face=BLUE_LIGHT, edge="#BFD4E4")
    fig.text(0.05, 0.404, "B", color="white", fontsize=8.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", facecolor=BLUE, edgecolor=BLUE))
    fig.text(0.077, 0.406, "A checkable counterfactual", color=INK, fontsize=10.2, fontweight="bold")
    fig.text(0.055, 0.352, "Fixed conversion rule", color=MUTED, fontsize=7.4, fontweight="bold")
    fig.text(0.055, 0.312, "F = (9/5) C + 32", color=INK, fontsize=14.0, fontweight="bold")
    fig.text(0.055, 0.271, "30.0 C = 86.0 F", color=GREEN, fontsize=12.2, fontweight="bold")

    add_round_box(fig, (0.055, 0.186, 0.126, 0.056), face=RED_LIGHT, edge="#E9B8B8", radius=0.008)
    add_round_box(fig, (0.191, 0.186, 0.126, 0.056), face=GREEN_LIGHT, edge="#B6DACD", radius=0.008)
    fig.text(0.118, 0.218, "FALSE", color=RED, fontsize=7.2, fontweight="bold", ha="center")
    fig.text(0.118, 0.197, "AIR 68.0 F", color=INK, fontsize=8.1, fontweight="bold", ha="center")
    fig.text(0.254, 0.218, "CORRECTED", color=GREEN, fontsize=7.2, fontweight="bold", ha="center")
    fig.text(0.254, 0.197, "AIR 86.0 F", color=INK, fontsize=8.1, fontweight="bold", ha="center")
    fig.text(0.055, 0.147, "The false carrier is scene-compatible but mechanically wrong.", color=MUTED, fontsize=7.1)

    draw_sample_table(fig, sample["responses"])
    draw_aggregate(fig, rows)

    fig.text(0.035, 0.018,
             "Selected qualitative success (not prevalence)  |  Aggregate denominators are model-specific clean-correct populations  |  Renderer: deterministic scene-adaptive compositing",
             color=MUTED, fontsize=7.1)

    output_root.mkdir(parents=True, exist_ok=True)
    png_path = output_root / "figure5_scei_overview.png"
    pdf_path = output_root / "figure5_scei_overview.pdf"
    provenance_path = output_root / "figure5_scei_overview_provenance.json"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
        metadata={
            "Title": "SCEI-Images-300 construction and effect",
            "Author": "CausalTypographicAttack artifact generator",
            "Creator": "scripts/make_scei_figure5.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)

    provenance = {
        "schema_version": "cta/figure5-scei-overview-provenance-v1",
        "data": {"path": DATA_PATH.relative_to(ROOT).as_posix(), "sha256": sha256(DATA_PATH)},
        "aggregate": {"path": SUMMARY_PATH.relative_to(ROOT).as_posix(), "sha256": sha256(SUMMARY_PATH)},
        "images": [
            {"variant": variant, "path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
            for variant, path in image_paths.items()
        ],
        "outputs": [
            {"path": png_path.relative_to(ROOT).as_posix(), "sha256": sha256(png_path)},
            {"path": pdf_path.relative_to(ROOT).as_posix(), "sha256": sha256(pdf_path)},
        ],
        "selection_note": sample["selection_note"],
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return png_path, pdf_path, provenance_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    for path in make_figure(args.output_root.resolve()):
        print(path)


if __name__ == "__main__":
    main()
