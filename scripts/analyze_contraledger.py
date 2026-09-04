#!/usr/bin/env python3
"""Audit and aggregate complete ContraLedger model logs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.contraledger import (
    CONDITIONS,
    summarize,
    summarize_prior_adjusted,
    summarize_source_prior,
)
from cta.question_bench import file_sha256


_FROZEN_INPUT_FIELDS = (
    "schema_version",
    "item_id",
    "condition",
    "cue_level",
    "truth",
    "family",
    "scenario_id",
    "target_label",
    "source_path",
    "source_sha256",
    "image_path",
    "image_sha256",
    "mask_path",
    "mask_sha256",
    "carrier_quad",
    "question_polarity",
    "status_line",
    "question",
    "option_order",
    "options",
    "correct_semantic",
    "correct_answer",
    "target_semantic",
    "target_answer",
    "registered_read_text",
    "probe_prompts",
    "knowledge_expected_semantic",
    "record",
    "scene_question",
    "question_generation_uses_victim_outputs",
    "victim_outputs_used_for_selection",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_frozen_manifest(rows: list[dict]) -> dict:
    """Fail closed if registered inputs no longer match the files on disk."""
    hash_cache: dict[str, str] = {}

    def checked_hash(value: object, expected: object, label: str) -> None:
        path = Path(str(value))
        if not path.is_file():
            raise ValueError(f"missing frozen {label}: {path}")
        resolved = str(path.resolve())
        actual = hash_cache.setdefault(resolved, file_sha256(path))
        if actual != str(expected):
            raise ValueError(f"frozen {label} hash mismatch: {path}")

    by_item: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("question_generation_uses_victim_outputs") is not False:
            raise ValueError("manifest question generation is not victim-output independent")
        if row.get("victim_outputs_used_for_selection") is not False:
            raise ValueError("manifest selection used victim outputs")
        checked_hash(row["source_path"], row["source_sha256"], "source")
        checked_hash(row["image_path"], row["image_sha256"], "attack image")
        checked_hash(row["mask_path"], row["mask_sha256"], "carrier mask")
        by_item.setdefault(str(row["item_id"]), []).append(row)

    for item_id, item_rows in by_item.items():
        if {str(row["condition"]) for row in item_rows} != set(CONDITIONS):
            raise ValueError(f"{item_id}: incomplete registered factorial conditions")
        if len({str(row["question"]) for row in item_rows}) != 1:
            raise ValueError(f"{item_id}: question differs across registered conditions")
        if len({str(row["source_sha256"]) for row in item_rows}) != 1:
            raise ValueError(f"{item_id}: source differs across registered conditions")
        quads = {json.dumps(row["carrier_quad"], sort_keys=True) for row in item_rows}
        if len(quads) != 1:
            raise ValueError(f"{item_id}: carrier geometry differs across conditions")
        if len({str(row["mask_sha256"]) for row in item_rows}) != 1:
            raise ValueError(f"{item_id}: carrier mask differs across conditions")

    return {
        "status": "passed",
        "rows": len(rows),
        "items": len(by_item),
        "files_hashed": len(hash_cache),
    }


def audit_prediction_inputs(model_name: str, predictions: list[dict], manifest: dict) -> None:
    """Verify that a model log embeds the exact frozen input for every condition."""
    for row in predictions:
        key = (str(row["item_id"]), str(row["condition"]))
        frozen = manifest[key]
        for field in _FROZEN_INPUT_FIELDS:
            if row.get(field) != frozen.get(field):
                raise ValueError(f"{model_name}/{key}: frozen field changed: {field}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-log", action="append", required=True, help="MODEL=PATH")
    parser.add_argument(
        "--source-prior-log", action="append", default=[], help="MODEL=PATH"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest_rows = read_jsonl(manifest_path)
    manifest = {(str(row["item_id"]), str(row["condition"])): row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("manifest has duplicate item-condition keys")

    manifest_sha256 = file_sha256(manifest_path)
    manifest_audit = audit_frozen_manifest(manifest_rows)

    evidence = {
        "schema_version": "cta/contraledger-evidence-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "manifest_audit": manifest_audit,
        "models": {},
        "claim_boundary": (
            "Values-only is the falsity-isolating attack. Authority is non-evaluative. Explicit conclusion "
            "is an inference-framing upper bound. EOR requires exact Read and correct independent Knowledge."
        ),
    }
    summary_rows = []
    family_rows = []
    prior_rows = []
    source_prior_paths = {
        model: Path(value).resolve()
        for model, value in (assignment.split("=", 1) for assignment in args.source_prior_log)
    }
    for assignment in args.model_log:
        model_name, log_value = assignment.split("=", 1)
        path = Path(log_value).resolve()
        rows = read_jsonl(path)
        indexed = {(str(row["item_id"]), str(row["condition"])): row for row in rows}
        if len(indexed) != len(rows) or set(indexed) != set(manifest):
            raise ValueError(f"{model_name}: incomplete or duplicate coverage")
        audit_prediction_inputs(model_name, rows, manifest)
        if any(row.get("decide_parsed") not in {"yes", "no"} for row in rows):
            raise ValueError(f"{model_name}: unparsed Decide output")
        if any(row.get("knowledge_parsed") not in {"yes", "no"} for row in rows):
            raise ValueError(f"{model_name}: unparsed Knowledge output")
        provenance_path = path.parent / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("status") != "complete" or provenance.get("completed_rows") != len(rows):
            raise ValueError(f"{model_name}: incomplete run provenance")
        if provenance.get("manifest_sha256") != manifest_sha256:
            raise ValueError(f"{model_name}: run provenance references a different manifest")
        summary = summarize(rows)
        prior_adjusted = None
        if model_name in source_prior_paths:
            prior_path = source_prior_paths[model_name]
            source_rows = read_jsonl(prior_path)
            prior_provenance_path = prior_path.parent / "provenance.json"
            prior_provenance = json.loads(prior_provenance_path.read_text(encoding="utf-8"))
            if prior_provenance.get("status") != "complete":
                raise ValueError(f"{model_name}: incomplete source-prior provenance")
            if prior_provenance.get("manifest_sha256") != manifest_sha256:
                raise ValueError(f"{model_name}: source-prior run references a different manifest")
            prior_adjusted = summarize_prior_adjusted(rows, source_rows)
            prior_summary = summarize_source_prior(source_rows)
            for prior_row in prior_adjusted:
                prior_rows.append({"model": model_name, **prior_row})
        evidence["models"][model_name] = {
            "log": str(path),
            "log_sha256": file_sha256(path),
            "provenance": str(provenance_path),
            "provenance_sha256": file_sha256(provenance_path),
            "summary": summary,
            "source_prior": (
                {
                    "log": str(prior_path),
                    "log_sha256": file_sha256(prior_path),
                    "provenance": str(prior_provenance_path),
                    "provenance_sha256": file_sha256(prior_provenance_path),
                    "summary": prior_summary,
                    "prior_adjusted": prior_adjusted,
                }
                if prior_adjusted is not None else None
            ),
        }
        for row in summary["cue_levels"]:
            summary_rows.append({"model": model_name, **row})
        for row in summary["families"]:
            family_rows.append({"model": model_name, **row})

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "contraledger_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "model_cue_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output / "model_family_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(family_rows[0]))
        writer.writeheader()
        writer.writerows(family_rows)
    if prior_rows:
        with (output / "model_source_prior_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(prior_rows[0]))
            writer.writeheader()
            writer.writerows(prior_rows)
    print(json.dumps({"models": list(evidence["models"]), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
