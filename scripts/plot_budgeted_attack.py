#!/usr/bin/env python3
"""Plot query-budget curves directly from audited analysis JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_analysis(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--analysis must be TITLE=PATH")
    title, path = value.split("=", 1)
    return title, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", action="append", type=parse_analysis, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, len(args.analysis), figsize=(4.2 * len(args.analysis), 3.1), squeeze=False)
    palette = ["#2C7FB8", "#F28E2B", "#59A14F", "#B07AA1", "#E15759"]
    for panel, (title, path) in enumerate(args.analysis):
        data = json.loads(path.resolve().read_text(encoding="utf-8"))
        rows = data["results"]
        models = list(dict.fromkeys(row["model"] for row in rows))
        axis = axes[0][panel]
        for index, model in enumerate(models):
            items = sorted((row for row in rows if row["model"] == model), key=lambda row: row["query_budget"])
            x = [row["query_budget"] for row in items]
            y = [100 * row["adaptive_strict_conditional_asr"] for row in items]
            low = [100 * row["adaptive_ci_low"] for row in items]
            high = [100 * row["adaptive_ci_high"] for row in items]
            color = palette[index % len(palette)]
            axis.plot(x, y, marker="o", linewidth=2, color=color, label=model)
            axis.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
            baseline = 100 * items[0]["legacy_baseline_strict_conditional_asr"]
            axis.plot([1, max(x)], [baseline, baseline], color=color, alpha=0.45, linestyle=":", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Query budget")
        axis.set_xticks(sorted({row["query_budget"] for row in rows}))
        axis.set_ylim(-2, 102)
        axis.grid(axis="y", alpha=0.2)
        if panel == 0:
            axis.set_ylabel("Strict clean-conditioned ASR (%)")
        else:
            axis.tick_params(labelleft=False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=min(4, len(labels)), frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    if output.suffix.lower() != ".pdf":
        figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
