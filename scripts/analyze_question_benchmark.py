#!/usr/bin/env python3
"""Validate complete public-benchmark logs and emit evidence-backed tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import CONDITIONS, file_sha256, summarize_question_rows


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def two_sided_exact_binomial(smaller: int, total: int) -> float:
    """Exact two-sided p-value for a fair Bernoulli discordance test."""
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(smaller + 1)) / (2 ** total)
    return min(1.0, 2 * tail)


def paired_exact_test(rows: list[dict], baseline: str, method: str, threshold: float) -> dict:
    by_condition = {
        condition: {row["question_id"]: row for row in rows if row["condition"] == condition}
        for condition in ("no_attack", baseline, method)
    }
    eligible = {
        qid for qid, row in by_condition["no_attack"].items()
        if float(row["answer_score"]) >= threshold
    }
    method_only = baseline_only = 0
    for qid in eligible:
        baseline_success = float(by_condition[baseline][qid]["answer_score"]) < threshold
        method_success = float(by_condition[method][qid]["answer_score"]) < threshold
        method_only += int(method_success and not baseline_success)
        baseline_only += int(baseline_success and not method_success)
    discordant = method_only + baseline_only
    return {
        "baseline": baseline, "method": method, "n_clean_correct": len(eligible),
        "method_only_successes": method_only, "baseline_only_successes": baseline_only,
        "exact_mcnemar_p": two_sided_exact_binomial(min(method_only, baseline_only), discordant),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-log", action="append", required=True, help="MODEL=predictions.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clean-correct-threshold", type=float, default=1.0)
    args = parser.parse_args()
    manifest = read_jsonl(args.manifest.resolve())
    expected = {(row["question_id"], row["condition"]) for row in manifest}
    manifest_conditions = {row["condition"] for row in manifest}
    if not set(CONDITIONS).issubset(manifest_conditions):
        missing = sorted(set(CONDITIONS) - manifest_conditions)
        raise ValueError(f"manifest is missing registered CTA conditions: {missing}")

    evidence = {
        "schema_version": "cta/question-benchmark-analysis-v1",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest.resolve()),
        "metric_boundary": "diagnostic normalized short-answer score; not an official VQAv2 or LingoQA leaderboard score",
        "models": {},
    }
    table_rows = []
    for assignment in args.model_log:
        if "=" not in assignment:
            raise ValueError("--model-log must have MODEL=PATH form")
        model_name, value = assignment.split("=", 1)
        log_path = Path(value).resolve()
        rows = read_jsonl(log_path)
        keys = {(row["question_id"], row["condition"]) for row in rows}
        if keys != expected or len(rows) != len(expected):
            raise ValueError(f"{model_name}: incomplete or duplicate log ({len(rows)}/{len(expected)} rows)")
        provenance_path = log_path.parent / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("status") != "complete" or provenance.get("completed_rows") != len(expected):
            raise ValueError(f"{model_name}: provenance does not attest a complete run")
        summary = summarize_question_rows(rows, args.clean_correct_threshold)
        baselines = [
            condition for condition in (
                "naive_typography", "scene_coherent", "causal_direct",
                "rio_typography_easy", "rio_typography_medium", "rio_typography_hard",
            ) if condition in manifest_conditions
        ]
        tests = [paired_exact_test(
            rows, baseline, "evidence_cta", args.clean_correct_threshold,
        ) for baseline in baselines]
        evidence["models"][model_name] = {
            "log": str(log_path), "log_sha256": file_sha256(log_path),
            "provenance_sha256": file_sha256(provenance_path),
            "summary": summary, "paired_tests": tests,
        }
        for row in summary:
            table_rows.append({"model": model_name, **row})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "question_benchmark_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8",
    )
    with (args.output_dir / "question_benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Model & Condition & $n$ & Acc. & ASR & Target ASR \\\\",
        "\\midrule",
    ]
    for row in table_rows:
        def pct(value):
            return "--" if value is None else f"{100 * value:.1f}"
        lines.append(
            f"{row['model']} & {row['condition'].replace('_', ' ')} & {row['n_clean_correct']} & "
            f"{pct(row['diagnostic_accuracy'])} & {pct(row['clean_conditioned_asr'])} & "
            f"{pct(row['targeted_asr'])} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    (args.output_dir / "generated_question_benchmark_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )
    print(json.dumps({"models": list(evidence["models"]), "rows": len(table_rows)}, indent=2))


if __name__ == "__main__":
    main()
