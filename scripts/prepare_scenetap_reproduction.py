#!/usr/bin/env python3
"""Stage a frozen RIO subset for the complete SceneTAP pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    source_manifest = args.manifest.resolve()
    output_root = args.output_root.resolve()
    questions_path = output_root / "questions.json"
    source_rows_path = output_root / "source_rows.jsonl"
    if questions_path.exists() or source_rows_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable stage: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    selected = []
    seen = set()
    for row in read_jsonl(source_manifest):
        qid = str(row["question_id"])
        if row["condition"] != "no_attack" or qid in seen:
            continue
        selected.append(row)
        seen.add(qid)
        if len(selected) == args.limit:
            break
    if len(selected) != args.limit:
        raise ValueError(f"requested {args.limit} rows, found {len(selected)}")

    questions = []
    with source_rows_path.open("w", encoding="utf-8") as source_handle:
        for index, row in enumerate(selected):
            source = Path(row["image_path"]).resolve()
            name = f"rio_{index:04d}_{row['question_id']}.jpg"
            destination = output_root / "images" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied_hash = sha256(destination)
            if copied_hash != sha256(source):
                raise AssertionError("staged image hash mismatch")
            answers = row.get("answers", [])
            correct = row.get("correct_content") or (answers[0] if answers else "")
            record = {
                "question_id": str(row["question_id"]),
                "image": name,
                "text": row["question"],
                "answer": correct,
                "category": "RIO-Bench Obj-MC",
                "target_text": row.get("target_content", ""),
                "target_answer": row.get("target_answer", ""),
                "choices": row.get("choices"),
                "source_manifest_sha256": sha256(source_manifest),
                "source_image_sha256": copied_hash,
            }
            questions.append(record)
            source_handle.write(json.dumps({**row, "staged_image_path": str(destination)}, ensure_ascii=False, sort_keys=True) + "\n")
    questions_path.write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": "cta/scenetap-reproduction-stage-v1",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "questions": len(questions),
        "questions_path": str(questions_path),
        "questions_sha256": sha256(questions_path),
        "planner_boundary": "The official GPT-4o endpoint was unavailable; the reproduction uses a local Qwen2.5-VL-7B placement planner and must be labeled accordingly.",
    }
    (output_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
