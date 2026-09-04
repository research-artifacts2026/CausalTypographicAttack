#!/usr/bin/env python3
"""Fail-closed aggregation for the frozen three-state ContraLedger protocol."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.contraledger import neutral_record
from cta.contraledger_threeway import CONDITIONS, option_for_semantic, summarize
from cta.question_bench import file_sha256
from cta.scei_attack import CounterfactualRecord, registered_evidence_text


_FROZEN_FIELDS = (
    "schema_version",
    "item_id",
    "dataset",
    "family",
    "scenario_id",
    "target_label",
    "source_path",
    "source_sha256",
    "condition",
    "record_truth",
    "image_path",
    "image_sha256",
    "question",
    "option_map",
    "option_permutation_index",
    "knowledge_option_order",
    "correct_semantic",
    "correct_answer",
    "target_semantic",
    "target_answer",
    "registered_read_text",
    "probe_prompts",
    "record",
    "question_generation_uses_victim_outputs",
    "victim_outputs_used_for_selection",
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_manifest(rows: list[dict]) -> dict:
    indexed = {(str(row["item_id"]), str(row["condition"])): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate manifest item-condition key")
    item_ids = sorted({key[0] for key in indexed})
    expected = {(item_id, condition) for item_id in item_ids for condition in CONDITIONS}
    if set(indexed) != expected:
        raise ValueError("incomplete three-way manifest coverage")

    hash_cache: dict[str, str] = {}

    def checked(path_value: object, expected_hash: object, label: str) -> None:
        path = Path(str(path_value))
        if not path.is_file():
            raise ValueError(f"missing {label}: {path}")
        resolved = str(path.resolve())
        actual = hash_cache.setdefault(resolved, file_sha256(path))
        if actual != str(expected_hash):
            raise ValueError(f"{label} hash mismatch: {path}")

    for item_id in item_ids:
        triplet = [indexed[(item_id, condition)] for condition in CONDITIONS]
        if any(row.get("question_generation_uses_victim_outputs") is not False for row in triplet):
            raise ValueError(f"{item_id}: question generation used victim output")
        if any(row.get("victim_outputs_used_for_selection") is not False for row in triplet):
            raise ValueError(f"{item_id}: selection used victim output")
        if len({str(row["question"]) for row in triplet}) != 1:
            raise ValueError(f"{item_id}: question differs across conditions")
        if len({str(row["source_sha256"]) for row in triplet}) != 1:
            raise ValueError(f"{item_id}: source differs across conditions")
        if len({json.dumps(row["record"], sort_keys=True) for row in triplet}) != 1:
            raise ValueError(f"{item_id}: symbolic record differs across conditions")
        if [row["correct_semantic"] for row in triplet] != [
            "absent", "consistent", "inconsistent"
        ]:
            raise ValueError(f"{item_id}: invalid registered semantic answers")
        source, true_row, false_row = triplet
        record = neutral_record(CounterfactualRecord(**dict(source["record"])))
        if true_row["registered_read_text"] != registered_evidence_text(record, "true"):
            raise ValueError(f"{item_id}: true registered text disagrees with symbolic record")
        if false_row["registered_read_text"] != registered_evidence_text(record, "false"):
            raise ValueError(f"{item_id}: false registered text disagrees with symbolic record")
        for row in triplet:
            if row["correct_answer"] != option_for_semantic(row["option_map"], row["correct_semantic"]):
                raise ValueError(f"{item_id}: registered option does not match semantic answer")
        if false_row["target_answer"] != option_for_semantic(false_row["option_map"], "consistent"):
            raise ValueError(f"{item_id}: false-record target is not CONSISTENT")
        checked(source["source_path"], source["source_sha256"], "source image")
        if source["image_sha256"] != source["source_sha256"]:
            raise ValueError(f"{item_id}: source condition is not the registered source image")
        checked(source["image_path"], source["image_sha256"], "source condition image")
        for rendered in (true_row, false_row):
            checked(rendered["image_path"], rendered["image_sha256"], "rendered image")
            checked(rendered["mask_path"], rendered["mask_sha256"], "carrier mask")
        if true_row["carrier_quad"] != false_row["carrier_quad"]:
            raise ValueError(f"{item_id}: carrier geometry differs across twins")
        if true_row["mask_sha256"] != false_row["mask_sha256"]:
            raise ValueError(f"{item_id}: carrier mask differs across twins")
        if true_row["registered_read_text"] == false_row["registered_read_text"]:
            raise ValueError(f"{item_id}: true and false record text are identical")

    return {
        "status": "passed",
        "items": len(item_ids),
        "rows": len(rows),
        "files_hashed": len(hash_cache),
    }


def audit_predictions(model: str, rows: list[dict], manifest: dict[tuple[str, str], dict]) -> None:
    indexed = {(str(row["item_id"]), str(row["condition"])): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(manifest):
        raise ValueError(f"{model}: incomplete or duplicate prediction coverage")
    for key, row in indexed.items():
        frozen = manifest[key]
        for field in _FROZEN_FIELDS:
            if row.get(field) != frozen.get(field):
                raise ValueError(f"{model}/{key}: frozen field changed: {field}")
        if row.get("decide_parsed") not in {"consistent", "inconsistent", "absent"}:
            raise ValueError(f"{model}/{key}: unparsed Decide output")
        if key[1] == "record_false":
            if not isinstance(row.get("read_match"), bool):
                raise ValueError(f"{model}/{key}: missing Read audit")
            if row.get("knowledge_parsed") not in {"yes", "no"}:
                raise ValueError(f"{model}/{key}: unparsed Knowledge output")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-log", action="append", required=True, help="MODEL=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest_rows = read_jsonl(manifest_path)
    manifest = {
        (str(row["item_id"]), str(row["condition"])): row for row in manifest_rows
    }
    manifest_audit = audit_manifest(manifest_rows)
    manifest_hash = file_sha256(manifest_path)
    evidence = {
        "schema_version": "cta/contraledger-threeway-evidence-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "manifest_audit": manifest_audit,
        "primary_endpoint": (
            "false-record CONSISTENT target ASR conditioned on correct source-ABSENT "
            "and true-record-CONSISTENT controls"
        ),
        "claim_boundary": (
            "Controlled digital values-only carrier. EOR additionally requires exact "
            "false-record transcription and correct independent rule rejection."
        ),
        "models": {},
    }
    model_rows = []
    family_rows = []
    for assignment in args.model_log:
        model, path_value = assignment.split("=", 1)
        path = Path(path_value).resolve()
        rows = read_jsonl(path)
        audit_predictions(model, rows, manifest)
        provenance_path = path.parent / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("status") != "complete" or provenance.get("completed_rows") != len(rows):
            raise ValueError(f"{model}: run provenance is incomplete")
        if provenance.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"{model}: run references a different manifest")
        summary = summarize(rows)
        evidence["models"][model] = {
            "log": str(path),
            "log_sha256": file_sha256(path),
            "provenance": str(provenance_path),
            "provenance_sha256": file_sha256(provenance_path),
            "summary": summary,
        }
        model_rows.append({
            "model": model,
            **{key: value for key, value in summary.items() if key != "families"},
        })
        family_rows.extend({"model": model, **row} for row in summary["families"])

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "contraledger_threeway_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "model_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(model_rows[0]))
        writer.writeheader()
        writer.writerows(model_rows)
    with (output / "family_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(family_rows[0]))
        writer.writeheader()
        writer.writerows(family_rows)
    print(json.dumps({"models": list(evidence["models"]), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
