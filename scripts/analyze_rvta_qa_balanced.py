#!/usr/bin/env python3
"""Validate balanced RVTA-QA logs and generate JSON/CSV/LaTeX evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import file_sha256
from cta.rvta_qa_balanced import CONDITIONS, _condition_summary, summarize


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.1f}"


def exact_mcnemar_p(challenger_only: int, baseline_only: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes."""
    discordant = challenger_only + baseline_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(challenger_only, baseline_only) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def add_bias_diagnostics(rows: list[dict], summary: dict) -> dict:
    """Add diagnostics without changing the preregistered pooled primary metric.

    Clean conditioning can change the mixture of counterbalance cells when a
    victim has a strong answer-format or truth-direction bias.  We therefore
    report an equal-weight six-cell macro estimate and two truth-direction
    summaries alongside (not instead of) the pooled estimate.
    """
    metric_names = (
        "answer_accuracy",
        "read_accuracy",
        "clean_conditioned_target_asr",
        "grounded_clean_conditioned_asr",
    )
    macro = []
    for condition in CONDITIONS:
        cell_rows = [
            next(row for row in stratum["conditions"] if row["condition"] == condition)
            for stratum in summary["strata"]
        ]
        record = {"condition": condition, "n_strata": len(cell_rows)}
        for metric in metric_names:
            values = [row[metric] for row in cell_rows if row.get(metric) is not None]
            record[metric] = statistics.fmean(values) if values else None
            record[f"n_strata_{metric}"] = len(values)
        record["min_clean_correct_per_stratum"] = min(row["n_clean_correct"] for row in cell_rows)
        record["max_clean_correct_per_stratum"] = max(row["n_clean_correct"] for row in cell_rows)
        macro.append(record)

    clean = {row["item_id"]: row for row in rows if row["condition"] == "no_attack"}
    eligible = {
        item_id for item_id, row in clean.items()
        if row.get("parsed_semantic") == row.get("correct_semantic")
    }
    directions = []
    for truth in ("false", "true"):
        direction_rows = [
            row for row in rows if row["counterbalance_cell"].split(":", 1)[0] == truth
        ]
        direction_eligible = {
            item_id for item_id in eligible
            if clean[item_id]["counterbalance_cell"].split(":", 1)[0] == truth
        }
        directions.append({
            "proposition_truth": truth,
            "target_direction": "false_to_yes" if truth == "false" else "true_to_no",
            "n_clean_correct": len(direction_eligible),
            "conditions": [
                _condition_summary(direction_rows, condition, direction_eligible)
                for condition in CONDITIONS
            ],
        })
    indexed = {
        (row["item_id"], row["condition"]): row
        for row in rows if row["item_id"] in eligible
    }
    comparisons = []
    for challenger, baseline in (
        ("evidence_cta", "plain_claim"),
        ("causal_bridge", "plain_claim"),
        ("causal_bridge", "evidence_cta"),
    ):
        challenger_only = 0
        baseline_only = 0
        challenger_successes = 0
        baseline_successes = 0
        for item_id in sorted(eligible):
            challenger_row = indexed[(item_id, challenger)]
            baseline_row = indexed[(item_id, baseline)]
            challenger_ok = (
                challenger_row.get("parsed_semantic") == challenger_row.get("target_semantic")
                and bool(challenger_row.get("read_match"))
            )
            baseline_ok = (
                baseline_row.get("parsed_semantic") == baseline_row.get("target_semantic")
                and bool(baseline_row.get("read_match"))
            )
            challenger_successes += challenger_ok
            baseline_successes += baseline_ok
            challenger_only += challenger_ok and not baseline_ok
            baseline_only += baseline_ok and not challenger_ok
        comparisons.append({
            "challenger": challenger,
            "baseline": baseline,
            "n_clean_correct": len(eligible),
            "challenger_successes": challenger_successes,
            "baseline_successes": baseline_successes,
            "paired_difference": (
                (challenger_successes - baseline_successes) / len(eligible)
                if eligible else None
            ),
            "challenger_only": challenger_only,
            "baseline_only": baseline_only,
            "mcnemar_exact_two_sided_p": exact_mcnemar_p(challenger_only, baseline_only),
            "analysis_status": "exploratory_secondary",
        })
    summary["macro_six_cell"] = macro
    summary["truth_direction"] = directions
    summary["paired_grounded_tests"] = comparisons
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-log", action="append", required=True, help="MODEL=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = read_jsonl(manifest_path)
    expected = {(row["item_id"], row["condition"]) for row in manifest}
    if {row["condition"] for row in manifest} != set(CONDITIONS):
        raise ValueError("manifest condition set differs from balanced-v1")
    image_hashes = {(row["item_id"], row["condition"]): row["image_sha256"] for row in manifest}
    evidence = {
        "schema_version": "cta/rvta-qa-balanced-analysis-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "items": len({row["item_id"] for row in manifest}),
        "models": {},
        "metric": "semantic grounded clean-conditioned target ASR",
        "reporting_boundary": "balanced-v1 is an internal protocol and is not a public-benchmark SOTA claim",
    }
    table_rows = []
    for assignment in args.model_log:
        if "=" not in assignment:
            raise ValueError("--model-log must use MODEL=PATH")
        model_name, value = assignment.split("=", 1)
        log_path = Path(value).resolve()
        rows = read_jsonl(log_path)
        keys = {(row["item_id"], row["condition"]) for row in rows}
        if keys != expected or len(rows) != len(expected):
            raise ValueError(f"{model_name}: incomplete or duplicate log")
        for row in rows:
            key = (row["item_id"], row["condition"])
            if row["image_sha256"] != image_hashes[key] or file_sha256(row["image_path"]) != image_hashes[key]:
                raise ValueError(f"{model_name}: image hash mismatch for {key}")
        provenance_path = log_path.parent / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("status") != "complete" or provenance.get("completed_rows") != len(expected):
            raise ValueError(f"{model_name}: incomplete provenance")
        summary = add_bias_diagnostics(rows, summarize(rows))
        by_condition = {row["condition"]: row for row in summary["pooled"]}
        macro_by_condition = {row["condition"]: row for row in summary["macro_six_cell"]}
        evidence["models"][model_name] = {
            "log": str(log_path),
            "log_sha256": file_sha256(log_path),
            "provenance": str(provenance_path),
            "summary": summary,
        }
        table_rows.append({"model": model_name, "pooled": by_condition, "macro": macro_by_condition})

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "rvta_qa_balanced_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "rvta_qa_balanced_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["model", "condition", "n_clean_correct", "answer_accuracy", "read_accuracy",
                  "clean_conditioned_target_asr", "grounded_clean_conditioned_asr"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model_name, model in evidence["models"].items():
            for row in model["summary"]["pooled"]:
                writer.writerow({"model": model_name, **{key: row.get(key) for key in fields if key != "model"}})
    lines = [
        "\\begin{tabular}{lrrrrrr}", "\\toprule",
        "Model & Clean & Benign & Direct & Plain-G & Evidence-G & Bridge-G \\\\",
        "\\midrule",
    ]
    for row in table_rows:
        pooled = row["pooled"]
        lines.append(
            f"{row['model']} & {pct(pooled['no_attack']['answer_accuracy'])} & "
            f"{pct(pooled['benign_control']['clean_conditioned_target_asr'])} & "
            f"{pct(pooled['direct_answer']['clean_conditioned_target_asr'])} & "
            f"{pct(pooled['plain_claim']['grounded_clean_conditioned_asr'])} & "
            f"{pct(pooled['evidence_cta']['grounded_clean_conditioned_asr'])} & "
            f"{pct(pooled['causal_bridge']['grounded_clean_conditioned_asr'])} \\\\" 
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    (output / "generated_rvta_qa_balanced_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    macro_lines = [
        "\\begin{tabular}{llrrrr}", "\\toprule",
        "Model & Aggregation & Direct & Plain-G & Evidence-G & Bridge-G \\\\",
        "\\midrule",
    ]
    for row in table_rows:
        for label, key in (("Pooled", "pooled"), ("6-cell macro", "macro")):
            values = row[key]
            macro_lines.append(
                f"{row['model']} & {label} & "
                f"{pct(values['direct_answer']['clean_conditioned_target_asr'])} & "
                f"{pct(values['plain_claim']['grounded_clean_conditioned_asr'])} & "
                f"{pct(values['evidence_cta']['grounded_clean_conditioned_asr'])} & "
                f"{pct(values['causal_bridge']['grounded_clean_conditioned_asr'])} \\\\"
            )
    macro_lines.extend(("\\bottomrule", "\\end{tabular}"))
    (output / "generated_rvta_qa_balanced_macro_table.tex").write_text(
        "\n".join(macro_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"models": list(evidence["models"]), "items": evidence["items"]}, indent=2))


if __name__ == "__main__":
    main()
