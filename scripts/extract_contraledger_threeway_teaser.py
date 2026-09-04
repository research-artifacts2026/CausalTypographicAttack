#!/usr/bin/env python3
"""Extract one fully auditable qualitative ContraLedger attack trace."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import file_sha256
from scripts.analyze_contraledger_threeway import (
    audit_manifest,
    audit_predictions,
    read_jsonl,
)


CONDITION_FILENAMES = {
    "source_absent": "source.jpg",
    "record_true": "record_true.jpg",
    "record_false": "record_false.jpg",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-label", default="Qwen2.5-VL-3B")
    parser.add_argument("--item-id")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    log_path = args.model_log.resolve()
    manifest_rows = read_jsonl(manifest_path)
    manifest = {
        (str(row["item_id"]), str(row["condition"])): row for row in manifest_rows
    }
    audit_manifest(manifest_rows)
    predictions = read_jsonl(log_path)
    audit_predictions(args.model_label, predictions, manifest)
    by_item: dict[str, dict[str, dict]] = {}
    for row in predictions:
        by_item.setdefault(str(row["item_id"]), {})[str(row["condition"])] = row

    if args.item_id:
        chosen_id = args.item_id
        selection_rule = "author-registered item identifier"
    else:
        eligible = []
        for item_id, rows in by_item.items():
            if (
                rows["source_absent"].get("decide_parsed") == "absent"
                and rows["record_true"].get("decide_parsed") == "consistent"
                and rows["record_false"].get("decide_parsed") == "consistent"
                and rows["record_false"].get("read_match") is True
                and rows["record_false"].get("knowledge_parsed") == "no"
            ):
                eligible.append(item_id)
        if not eligible:
            raise ValueError("no complete EOR-success trace is available")
        chosen_id = sorted(eligible)[0]
        selection_rule = (
            "lexicographically first complete trace satisfying both controls, false target, "
            "exact read, and correct independent rejection; qualitative outcome-selected example"
        )
    if chosen_id not in by_item:
        raise ValueError(f"unknown item id: {chosen_id}")

    rows = by_item[chosen_id]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    image_records = {}
    for condition, filename in CONDITION_FILENAMES.items():
        source = Path(rows[condition]["image_path"])
        target = output / filename
        shutil.copy2(source, target)
        copied_hash = file_sha256(target)
        if copied_hash != rows[condition]["image_sha256"]:
            raise ValueError(f"copied {condition} image hash mismatch")
        image_records[condition] = {
            "path": filename,
            "sha256": copied_hash,
        }

    false = rows["record_false"]
    trace = {
        "schema_version": "cta/contraledger-threeway-teaser-v1",
        "model": args.model_label,
        "item_id": chosen_id,
        "family": false["family"],
        "selection_rule": selection_rule,
        "claim_boundary": (
            "One qualitative logged digital example; outcome-selected when --item-id is omitted. "
            "It is not a prevalence estimate or camera capture."
        ),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "model_log": str(log_path),
        "model_log_sha256": file_sha256(log_path),
        "run_provenance_sha256": file_sha256(log_path.parent / "provenance.json"),
        "question": false["question"],
        "option_map": false["option_map"],
        "registered_false_record": false["registered_read_text"],
        "carrier_quad": false["carrier_quad"],
        "images": image_records,
        "responses": {
            condition: {
                "raw": rows[condition]["decide_raw"],
                "parsed": rows[condition]["decide_parsed"],
                "correct_semantic": rows[condition]["correct_semantic"],
            }
            for condition in CONDITION_FILENAMES
        },
        "read_raw": false["read_raw"],
        "read_match": false["read_match"],
        "knowledge_prompt": false["probe_prompts"]["knowledge"],
        "knowledge_raw": false["knowledge_raw"],
        "knowledge_parsed": false["knowledge_parsed"],
    }
    (output / "trace.json").write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"item_id": chosen_id, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
