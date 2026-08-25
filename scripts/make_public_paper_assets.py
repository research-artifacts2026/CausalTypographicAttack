#!/usr/bin/env python3
"""Generate public RIO and violation-severity paper assets from audited evidence.

The script intentionally consumes only aggregate evidence JSON written by the
analysis programs.  It never reads hand-entered paper numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


MODEL_ORDER = (
    "Qwen2.5-VL-3B",
    "Qwen2.5-VL-7B",
    "LLaVA-OneVision-1.5-8B",
    "InternVL2-8B",
)

MODEL_LABELS = {
    "Qwen2.5-VL-3B": "Qwen2.5-VL-3B",
    "Qwen2.5-VL-7B": "Qwen2.5-VL-7B",
    "LLaVA-OneVision-1.5-8B": "LLaVA-OV-1.5-8B",
    "InternVL2-8B": "InternVL2-8B",
}

CONDITION_LABELS = (
    ("naive_typography", "Naive"),
    ("rio_typography_hard", "RIO-hard"),
    ("rio_scenetap_hard", "SceneTAP"),
    ("cta_identity_card", "CTA-v2"),
)

SEVERITY_ORDER = ("moderate", "strong", "extreme")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latex_escape(value: str) -> str:
    return (
        value.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def pct(value: float, digits: int = 1) -> str:
    return ("{:0." + str(digits) + "f}").format(100.0 * value)


def index_summary(model: dict) -> dict:
    return {row["condition"]: row for row in model["summary"]}


def make_rio_table(evidence: dict, output: Path) -> None:
    rows = []
    method_means = {condition: [] for condition, _ in CONDITION_LABELS}
    clean_means = []
    for model_name in MODEL_ORDER:
        summary = index_summary(evidence["models"][model_name])
        clean = summary["no_attack"]
        clean_means.append(clean["diagnostic_accuracy"])
        values = []
        for condition, _ in CONDITION_LABELS:
            value = summary[condition]["clean_conditioned_asr"]
            method_means[condition].append(value)
            values.append(pct(value))
        rows.append(
            "{} & {} & {} & {} \\\\".format(
                latex_escape(MODEL_LABELS[model_name]),
                clean["n_clean_correct"],
                pct(clean["diagnostic_accuracy"]),
                " & ".join(values),
            )
        )
    macro = [sum(method_means[c]) / len(method_means[c]) for c, _ in CONDITION_LABELS]
    rows.append(r"\midrule")
    rows.append(
        r"Macro mean & -- & {:.1f} & {} \\".format(
            100.0 * sum(clean_means) / len(clean_means),
            " & ".join("{:.1f}".format(100.0 * value) for value in macro),
        )
    )
    header = "Model & $n_c$ & Clean & " + " & ".join(label for _, label in CONDITION_LABELS) + r" \\"
    text = "\n".join(
        [r"\begin{tabular}{lrrrrrr}", r"\toprule", header, r"\midrule"]
        + rows
        + [r"\bottomrule", r"\end{tabular}", ""]
    )
    output.write_text(text, encoding="utf-8")


def severity_rows(evidence: dict) -> list[dict]:
    summaries = evidence["models"]["Qwen2.5-VL-3B"]["summaries"]
    indexed = {
        row["value"]: row
        for row in summaries
        if row["dimension"] == "severity"
    }
    return [indexed[level] for level in SEVERITY_ORDER]


def make_severity_table(evidence: dict, output: Path) -> None:
    body = []
    for row in severity_rows(evidence):
        interval = "[{}, {}]".format(
            pct(row["asr_wilson95_low"]), pct(row["asr_wilson95_high"])
        )
        body.append(
            "{} & {} & {} & {} & {} \\\\".format(
                row["value"].capitalize(),
                row["n"],
                pct(row["strict_false_claim_acceptance_asr"]),
                interval,
                pct(row["grounded_transcription_rate"]),
            )
        )
    text = "\n".join(
        [
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Severity & $n$ & Strict ASR & Wilson 95\% CI & Read \\",
            r"\midrule",
        ]
        + body
        + [r"\bottomrule", r"\end{tabular}", ""]
    )
    output.write_text(text, encoding="utf-8")


def make_severity_figure(evidence: dict, output: Path) -> None:
    summaries = evidence["models"]["Qwen2.5-VL-3B"]["summaries"]
    scenario_rows = {}
    for row in summaries:
        if row["dimension"] != "scenario_severity":
            continue
        scenario, severity = row["value"].split("::", 1)
        scenario_rows.setdefault(scenario, {})[severity] = row

    scenarios = (
        ("seventy-degree-freshness", "Food at 70 C", "#4477AA", "o", "-"),
        ("thermal-exposure", "Person at 70--300 C", "#EE6677", "s", "--"),
        ("unaided-flight", "Unaided human flight", "#228833", "^", "-."),
        ("zero-energy-travel", "Zero-energy travel", "#CCBB44", "D", ":"),
        ("zero-mass-matter", "Zero-mass object", "#AA3377", "v", "-"),
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = list(range(len(SEVERITY_ORDER)))
    for scenario, label, color, marker, line_style in scenarios:
        values = scenario_rows[scenario]
        rates = [100.0 * values[level]["strict_false_claim_acceptance_asr"] for level in SEVERITY_ORDER]
        counts = [values[level]["n"] for level in SEVERITY_ORDER]
        count_label = "n={}/level".format(counts[0]) if len(set(counts)) == 1 else "n={}".format(counts)
        ax.plot(
            x,
            rates,
            label="{} ({})".format(label, count_label),
            color=color,
            marker=marker,
            linestyle=line_style,
            linewidth=2.1,
            markersize=6,
        )
    fig.suptitle(
        "Greater absurdity does not reliably increase attack success",
        x=0.11,
        y=0.965,
        ha="left",
        fontsize=13,
        weight="bold",
    )
    fig.text(
        0.11,
        0.895,
        "Qwen2.5-VL-3B; matched severity levels within each predeclared scenario",
        fontsize=9.5,
        color="#555555",
        ha="left",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([level.capitalize() for level in SEVERITY_ORDER])
    ax.set_ylabel("Strict false-claim acceptance ASR (%)")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        handlelength=2.8,
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.83, bottom=0.30)
    fig.savefig(str(output), dpi=240, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rio-evidence", type=Path, required=True)
    parser.add_argument("--severity-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rio = load_json(args.rio_evidence)
    severity = load_json(args.severity_evidence)
    make_rio_table(rio, args.output_dir / "generated_rio_ctav2_table.tex")
    make_severity_table(severity, args.output_dir / "generated_severity_table.tex")
    make_severity_figure(severity, args.output_dir / "violation_severity_curve.png")


if __name__ == "__main__":
    main()
