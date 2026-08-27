#!/usr/bin/env python3
"""Majority-aggregate independent RVTA-Context source reviews."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import file_sha256


FIELDS = ("outdoor_scene", "location_credible", "carrier_region_approved")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--response", type=Path, action="append", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--minimum-annotators", type=int, default=3)
    args = parser.parse_args()
    source_manifest = args.source_manifest.resolve()
    source_rows = read_jsonl(source_manifest)
    source_hash = file_sha256(source_manifest)
    source_ids = {row["item_id"] for row in source_rows}
    responses = []
    annotators = set()
    for path in args.response:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("source_manifest_sha256") != source_hash:
            raise ValueError(f"{path}: source manifest hash mismatch")
        annotator = str(payload.get("annotator", ""))
        if not annotator or annotator in annotators:
            raise ValueError(f"{path}: missing or duplicate annotator")
        annotators.add(annotator)
        rows = payload.get("responses", [])
        if {row.get("item_id") for row in rows} != source_ids or len(rows) != len(source_ids):
            raise ValueError(f"{path}: incomplete response coverage")
        responses.append({row["item_id"]: row for row in rows})
    if len(responses) < args.minimum_annotators:
        raise ValueError("not enough independent response files")
    threshold = len(responses) // 2 + 1
    approved = []
    excluded = []
    for source in source_rows:
        item_id = source["item_id"]
        votes = {field: sum(bool(values[item_id].get(field)) for values in responses) for field in FIELDS}
        review = {field: votes[field] >= threshold for field in FIELDS}
        review["annotators"] = len(responses)
        review["vote_counts"] = votes
        notes = [values[item_id].get("note", "") for values in responses if values[item_id].get("note")]
        fully_approved = all(review[field] for field in FIELDS)
        review["exclude_reason"] = "" if fully_approved else "; ".join(notes) or "failed majority review"
        source["manual_review"] = review
        if fully_approved:
            approved.append(source)
        else:
            excluded.append({"item_id": item_id, "votes": votes, "notes": notes})
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as handle:
        for row in approved:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": "cta/rvta-context-review-analysis-v1",
        "source_manifest_sha256": source_hash,
        "annotators": len(responses),
        "majority_threshold": threshold,
        "source_items": len(source_rows),
        "approved_items": len(approved),
        "excluded_items": len(excluded),
        "excluded": excluded,
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": file_sha256(args.output_manifest),
    }
    report_path = args.output_manifest.with_suffix(".review.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("source_items", "approved_items", "excluded_items")}, indent=2))


if __name__ == "__main__":
    main()
