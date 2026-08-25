#!/usr/bin/env python3
"""Plot validated RVTA-QA grounded clean-conditioned ASR across datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ("causal_claim", "evidence_cta", "causal_bridge")
DISPLAY = {"causal_claim": "Claim", "evidence_cta": "Evidence", "causal_bridge": "Bridge"}
COLORS = {"causal_claim": "#56B4E9", "evidence_cta": "#E69F00", "causal_bridge": "#009E73"}


def load_assignment(value: str) -> tuple[str, dict]:
    if "=" not in value:
        raise ValueError("--evidence must use LABEL=PATH")
    label, path = value.split("=", 1)
    return label, json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", action="append", required=True, help="LABEL=validated evidence JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panels = [load_assignment(value) for value in args.evidence]
    if len(panels) not in (1, 2):
        raise ValueError("plot supports one or two dataset panels")

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    })
    figure, axes = plt.subplots(1, len(panels), figsize=(3.35 * len(panels), 2.15), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, (label, evidence) in zip(axes, panels):
        models = list(evidence["models"])
        x = np.arange(len(models))
        width = 0.23
        for index, condition in enumerate(CONDITIONS):
            values, low, high = [], [], []
            for model in models:
                summary = {
                    row["condition"]: row for row in evidence["models"][model]["summary"]
                }[condition]
                value = 100 * summary["grounded_clean_conditioned_asr"]
                values.append(value)
                # Matplotlib rejects tiny negative error lengths introduced by
                # floating-point roundoff at exact 0/100% endpoints.
                low.append(max(0.0, value - 100 * summary["grounded_wilson95_low"]))
                high.append(max(0.0, 100 * summary["grounded_wilson95_high"] - value))
            offset = (index - 1) * width
            axis.bar(
                x + offset, values, width, label=DISPLAY[condition], color=COLORS[condition],
                edgecolor="black", linewidth=0.35,
            )
            axis.errorbar(
                x + offset, values, yerr=np.array([low, high]), fmt="none",
                ecolor="black", elinewidth=0.55, capsize=1.8, capthick=0.55,
            )
        axis.set_title(label)
        axis.set_xticks(x)
        axis.set_xticklabels([name.replace("Qwen2.5-VL-", "Qwen-").replace("LLaVA-OneVision-1.5-8B", "LLaVA").replace("InternVL2-8B", "InternVL") for name in models], rotation=18, ha="right")
        axis.set_ylim(0, 100)
        axis.grid(axis="y", color="#d0d0d0", linewidth=0.45, alpha=0.7)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Grounded ASR (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    figure.savefig(args.output.with_suffix(".png"), dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    main()
