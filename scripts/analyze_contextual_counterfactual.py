#!/usr/bin/env python3
"""Aggregate complete RVTA-Context logs into provenance-linked evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.contextual_counterfactual import CONDITIONS, SEVERITIES, summarize


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-log", action="append", required=True, help="MODEL=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    evidence = {
        "schema_version": "cta/rvta-context-crossmodel-evidence-v1",
        "endpoint": "same-severity clean-conditioned target flip with exact main-claim transcription",
        "models": {},
    }
    table_rows = []
    for assignment in args.model_log:
        if "=" not in assignment:
            raise ValueError("--model-log must use MODEL=PATH")
        model, raw_path = assignment.split("=", 1)
        path = Path(raw_path).resolve()
        rows = read_jsonl(path)
        item_ids = {str(row["item_id"]) for row in rows}
        keys = {(str(row["item_id"]), row["condition"]) for row in rows}
        expected = {(item_id, condition) for item_id in item_ids for condition in CONDITIONS}
        if not rows or keys != expected or len(keys) != len(rows):
            raise ValueError(f"{model}: incomplete or duplicate log")
        provenance_path = path.parent / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("status") != "complete" or int(provenance.get("completed_rows", -1)) != len(rows):
            raise ValueError(f"{model}: incomplete run provenance")
        summary = summarize(rows)
        evidence["models"][model] = {
            "log": str(path),
            "log_sha256": sha256(path),
            "provenance": str(provenance_path),
            "provenance_sha256": sha256(provenance_path),
            "summary": summary,
        }
        indexed = {row["condition"]: row for row in summary["conditions"]}
        for severity in SEVERITIES:
            plain = indexed[f"false_{severity}_plain"]
            bridge = indexed[f"false_{severity}_bridge"]
            table_rows.append({
                "model": model,
                "severity": severity,
                "n_clean_eligible": bridge["n_clean_eligible"],
                "plain_grounded_asr": plain["grounded_clean_conditioned_asr"],
                "bridge_grounded_asr": bridge["grounded_clean_conditioned_asr"],
                "plain_value_capture": plain["false_value_capture_rate"],
                "bridge_value_capture": bridge["false_value_capture_rate"],
                "plain_read_rate": plain["exact_read_rate"],
                "bridge_read_rate": bridge["exact_read_rate"],
            })

    evidence["reporting_boundary"] = (
        "The Singapore slice tests scene-matched visual-text override against a stated trusted sensor record. "
        "It is not live-weather estimation, a physical capture, or a general multi-family benchmark."
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rvta_context_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "rvta_context_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    print(json.dumps({"models": list(evidence["models"]), "rows": len(table_rows)}, indent=2))


if __name__ == "__main__":
    main()
