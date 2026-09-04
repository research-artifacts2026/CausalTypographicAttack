#!/usr/bin/env python3
"""Audit and aggregate complete Attribute-CF-v1 model logs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.attribute_counterfactual import CONDITIONS, FAMILY_ORDER, summarize


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-log", action="append", required=True, help="MODEL=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    evidence = {
        "schema_version": "cta/attribute-counterfactual-evidence-v1",
        "primary_endpoint": "clean-and-true-plain-conditioned target decision rate",
        "mechanism_endpoint": "KDI after exact Read, correct Ground, and Verify rejection",
        "models": {},
    }
    model_rows = []
    family_rows = []
    for assignment in args.model_log:
        if "=" not in assignment:
            raise ValueError("--model-log must use MODEL=PATH")
        model_name, value = assignment.split("=", 1)
        path = Path(value).resolve()
        rows = read_jsonl(path)
        item_ids = {row["item_id"] for row in rows}
        keys = {(row["item_id"], row["condition"]) for row in rows}
        expected = {(item_id, condition) for item_id in item_ids for condition in CONDITIONS}
        if not rows or keys != expected or len(keys) != len(rows):
            raise ValueError(f"{model_name}: incomplete or duplicate condition coverage")
        if any(row.get("decide_parsed") not in {"yes", "no"} for row in rows):
            raise ValueError(f"{model_name}: unparsed Decide output present")
        provenance_path = path.parent / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("status") != "complete" or int(provenance.get("completed_rows", -1)) != len(rows):
            raise ValueError(f"{model_name}: provenance does not attest a complete run")
        summary = summarize(rows)
        evidence["models"][model_name] = {
            "log": str(path),
            "log_sha256": sha256(path),
            "provenance": str(provenance_path),
            "provenance_sha256": sha256(provenance_path),
            "summary": summary,
        }
        by_condition = {row["condition"]: row for row in summary["conditions"]}
        model_rows.append({
            "model": model_name,
            "items": summary["items"],
            "common_clean_correct": summary["n_common_clean_correct"],
            "true_plain_target_rate": by_condition["true_plain"]["target_decision_rate"],
            "false_plain_target_rate": by_condition["false_plain"]["target_decision_rate"],
            "true_conclusion_target_rate": by_condition["true_target_conclusion"]["target_decision_rate"],
            "false_conclusion_target_rate": by_condition["false_target_conclusion"]["target_decision_rate"],
            "counterfactual_effect": summary["factorial"]["counterfactual_effect_without_conclusion"],
            "interaction": summary["factorial"]["false_by_conclusion_interaction"],
            "false_plain_read": by_condition["false_plain"]["exact_read_rate"],
            "false_plain_verify_accuracy": by_condition["false_plain"]["verify_accuracy"],
            "false_plain_kdi_n": by_condition["false_plain"]["kdi_n"],
            "false_plain_kdi_rate": by_condition["false_plain"]["kdi_rate"],
        })
        for family in summary["families"]:
            family_rows.append({"model": model_name, **family})

    macros = {}
    numeric_fields = (
        "true_plain_target_rate", "false_plain_target_rate", "true_conclusion_target_rate",
        "false_conclusion_target_rate", "counterfactual_effect", "interaction",
        "false_plain_read", "false_plain_verify_accuracy", "false_plain_kdi_rate",
    )
    for field in numeric_fields:
        values = [row[field] for row in model_rows if row[field] is not None]
        macros[field] = sum(values) / len(values) if values else None
    evidence["macro_across_models"] = macros
    evidence["family_order"] = list(FAMILY_ORDER)
    evidence["reporting_boundary"] = (
        "The 120-item development pilot is frozen before victim inference. Target decision rate is not "
        "called grounded ASR unless exact Read and correct Ground are separately shown. KDI is descriptive "
        "unless its denominator is large enough, and no cross-model SOTA claim follows from two checkpoints."
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "attribute_counterfactual_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "model_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(model_rows[0]))
        writer.writeheader()
        writer.writerows(model_rows)
    with (args.output_dir / "family_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(family_rows[0]))
        writer.writeheader()
        writer.writerows(family_rows)
    print(json.dumps({"models": list(evidence["models"]), "macro": macros}, indent=2))


if __name__ == "__main__":
    main()
