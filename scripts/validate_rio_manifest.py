#!/usr/bin/env python3
"""Validate a materialized RIO manifest before any model queries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.run_validation import file_sha256, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--expected-questions", type=int, required=True)
    parser.add_argument("--expected-conditions", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    provenance_path = args.provenance.resolve()
    rows = read_jsonl(manifest_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("status") != "complete":
        raise ValueError("build provenance is not complete")
    if file_sha256(manifest_path) != provenance.get("manifest_sha256"):
        raise ValueError("manifest hash differs from build provenance")
    if int(provenance.get("rows", -1)) != len(rows):
        raise ValueError("manifest row count differs from build provenance")

    keys = [(str(row["question_id"]), str(row["condition"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("manifest contains duplicate question-condition keys")
    question_ids = sorted({qid for qid, _ in keys})
    conditions = sorted({condition for _, condition in keys})
    if len(question_ids) != args.expected_questions:
        raise ValueError(
            f"expected {args.expected_questions} questions, found {len(question_ids)}"
        )
    if len(conditions) != args.expected_conditions:
        raise ValueError(
            f"expected {args.expected_conditions} conditions, found {len(conditions)}"
        )
    expected_set = set(conditions)
    for qid in question_ids:
        present = {condition for row_qid, condition in keys if row_qid == qid}
        if present != expected_set:
            raise ValueError(f"{qid}: condition set differs from the registered set")

    reference = {}
    checked_images = set()
    for row in rows:
        qid = str(row["question_id"])
        stable = (row["question"], row["answers"], row.get("choices"))
        if qid in reference and reference[qid] != stable:
            raise ValueError(f"{qid}: question, gold answer, or choices differ by condition")
        reference[qid] = stable
        image_path = Path(row["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        key = (str(image_path), row["image_sha256"])
        if key not in checked_images:
            if file_sha256(image_path) != row["image_sha256"]:
                raise ValueError(f"image hash mismatch: {image_path}")
            checked_images.add(key)

    audit = {
        "schema_version": "cta/rio-manifest-audit-v1",
        "status": "complete",
        "questions": len(question_ids),
        "conditions": conditions,
        "rows": len(rows),
        "unique_images_checked": len(checked_images),
        "manifest_sha256": file_sha256(manifest_path),
        "resolved_revision": provenance.get("resolved_revision"),
    }
    text = json.dumps(audit, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
