#!/usr/bin/env python3
"""Validate balanced RVTA-QA logs and generate JSON/CSV/LaTeX evidence."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import file_sha256
from cta.rvta_qa_balanced import CONDITIONS, summarize


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.1f}"


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
        summary = summarize(rows)
        by_condition = {row["condition"]: row for row in summary["pooled"]}
        evidence["models"][model_name] = {
            "log": str(log_path),
            "log_sha256": file_sha256(log_path),
            "provenance": str(provenance_path),
            "summary": summary,
        }
        table_rows.append({"model": model_name, **by_condition})

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
        lines.append(
            f"{row['model']} & {pct(row['no_attack']['answer_accuracy'])} & "
            f"{pct(row['benign_control']['clean_conditioned_target_asr'])} & "
            f"{pct(row['direct_answer']['clean_conditioned_target_asr'])} & "
            f"{pct(row['plain_claim']['grounded_clean_conditioned_asr'])} & "
            f"{pct(row['evidence_cta']['grounded_clean_conditioned_asr'])} & "
            f"{pct(row['causal_bridge']['grounded_clean_conditioned_asr'])} \\\\" 
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    (output / "generated_rvta_qa_balanced_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"models": list(evidence["models"]), "items": evidence["items"]}, indent=2))


if __name__ == "__main__":
    main()

