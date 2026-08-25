#!/usr/bin/env python3
"""Create a paper figure only from validated balanced-analysis evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = (
    ("plain_claim", "Plain claim", "#7a7a7a"),
    ("evidence_cta", "Evidence CTA", "#2b6cb0"),
    ("causal_bridge", "Causal bridge", "#c05621"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_evidence(path: Path) -> dict:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("schema_version") != "cta/rvta-qa-balanced-analysis-v1":
        raise ValueError(f"unexpected evidence schema: {path}")
    if not evidence.get("models"):
        raise ValueError(f"evidence has no models: {path}")
    return evidence


def short_model(name: str) -> str:
    replacements = {
        "Qwen2.5-VL-3B": "Qwen-3B",
        "Qwen2.5-VL-7B": "Qwen-7B",
        "LLaVA-OneVision-7B": "LLaVA",
        "LLaVA-OneVision-1.5-8B": "LLaVA",
        "InternVL2-8B": "InternVL",
    }
    return replacements.get(name, name)


def macro_values(evidence: dict, condition: str) -> list[float]:
    values = []
    for model in evidence["models"].values():
        rows = {row["condition"]: row for row in model["summary"]["macro_six_cell"]}
        value = rows[condition]["grounded_clean_conditioned_asr"]
        if value is None:
            raise ValueError(f"missing macro grounded ASR for {condition}")
        values.append(100.0 * value)
    return values


def pct(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-evidence", type=Path, required=True)
    parser.add_argument("--voc-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    inputs = (("COCO", args.coco_evidence.resolve()), ("Pascal VOC", args.voc_evidence.resolve()))
    datasets = [(label, path, read_evidence(path)) for label, path in inputs]
    model_orders = [list(evidence["models"]) for _, _, evidence in datasets]
    if model_orders[0] != model_orders[1]:
        raise ValueError("COCO and VOC evidence use different model order")
    labels = [short_model(name) for name in model_orders[0]]

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), sharey=True)
    x = np.arange(len(labels))
    width = 0.23
    for axis, (dataset_label, _, evidence) in zip(axes, datasets):
        for index, (condition, condition_label, color) in enumerate(CONDITIONS):
            offset = (index - 1) * width
            values = macro_values(evidence, condition)
            bars = axis.bar(x + offset, values, width, label=condition_label, color=color)
            axis.bar_label(bars, fmt="%.0f", padding=1, fontsize=7)
        axis.set_title(f"{dataset_label} (n={evidence['items']})")
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=0.2, linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Six-cell macro grounded ASR (%)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False,
                  bbox_to_anchor=(0.5, 1.03))
    figure.tight_layout(rect=(0, 0, 1, 0.91))

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pdf_path = output / "rvta_qa_balanced_crossdataset.pdf"
    png_path = output / "rvta_qa_balanced_crossdataset.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    table_path = output / "generated_rvta_qa_balanced_crossdataset_table.tex"
    table_lines = [
        "\\begin{tabular}{llrrrrrrr}", "\\toprule",
        "Dataset & Model & $n_c$ & Clean & Benign & Plain-G & Evidence-G & Bridge-G & Bridge-M \\\\",
        "\\midrule",
    ]
    direction_path = output / "generated_rvta_qa_balanced_direction_table.tex"
    direction_lines = [
        "\\begin{tabular}{llrrrr}", "\\toprule",
        "Dataset & Model & $n_{F}$ & F$\\rightarrow$Y & $n_{T}$ & T$\\rightarrow$N \\\\",
        "\\midrule",
    ]
    for dataset_label, _, evidence in datasets:
        for model_name, model in evidence["models"].items():
            summary = model["summary"]
            pooled = {row["condition"]: row for row in summary["pooled"]}
            macro = {row["condition"]: row for row in summary["macro_six_cell"]}
            table_lines.append(
                f"{dataset_label} & {short_model(model_name)} & {summary['n_clean_correct']} & "
                f"{pct(pooled['no_attack']['answer_accuracy'])} & "
                f"{pct(pooled['benign_control']['clean_conditioned_target_asr'])} & "
                f"{pct(pooled['plain_claim']['grounded_clean_conditioned_asr'])} & "
                f"{pct(pooled['evidence_cta']['grounded_clean_conditioned_asr'])} & "
                f"{pct(pooled['causal_bridge']['grounded_clean_conditioned_asr'])} & "
                f"{pct(macro['causal_bridge']['grounded_clean_conditioned_asr'])} \\\\"
            )
            directions = {row["proposition_truth"]: row for row in summary["truth_direction"]}
            false_bridge = next(
                row for row in directions["false"]["conditions"] if row["condition"] == "causal_bridge"
            )
            true_bridge = next(
                row for row in directions["true"]["conditions"] if row["condition"] == "causal_bridge"
            )
            direction_lines.append(
                f"{dataset_label} & {short_model(model_name)} & {directions['false']['n_clean_correct']} & "
                f"{pct(false_bridge['grounded_clean_conditioned_asr'])} & "
                f"{directions['true']['n_clean_correct']} & "
                f"{pct(true_bridge['grounded_clean_conditioned_asr'])} \\\\"
            )
        table_lines.append("\\addlinespace")
        direction_lines.append("\\addlinespace")
    table_lines[-1] = "\\bottomrule"
    direction_lines[-1] = "\\bottomrule"
    table_lines.append("\\end{tabular}")
    direction_lines.append("\\end{tabular}")
    table_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    direction_path.write_text("\n".join(direction_lines) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": "cta/rvta-qa-balanced-figure-v1",
        "aggregation": "unweighted mean over six preregistered counterbalance cells",
        "metric": "grounded clean-conditioned target ASR",
        "inputs": [
            {"dataset": label, "path": str(path), "sha256": sha256(path)}
            for label, path in inputs
        ],
        "outputs": {
            "pdf": {"path": str(pdf_path), "sha256": sha256(pdf_path)},
            "png": {"path": str(png_path), "sha256": sha256(png_path)},
            "crossdataset_table": {"path": str(table_path), "sha256": sha256(table_path)},
            "direction_table": {"path": str(direction_path), "sha256": sha256(direction_path)},
        },
    }
    (output / "rvta_qa_balanced_figure_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance["outputs"], indent=2))


if __name__ == "__main__":
    main()
