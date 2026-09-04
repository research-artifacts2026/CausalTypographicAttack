#!/usr/bin/env python3
"""Stage a frozen three-state manifest for a matched SceneTAP replay.

SceneTAP plans the false record once per source image.  The chosen region is
then reused for the corrected twin so that the truth pair has identical
geometry.  Text and items are frozen before planning or victim inference.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.contraledger_threeway import CONDITIONS
from cta.question_bench import file_sha256
from scripts.analyze_contraledger_threeway import audit_manifest


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "item"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source_manifest = args.source_manifest.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable stage: {output}")
    rows = read_jsonl(source_manifest)
    manifest_audit = audit_manifest(rows)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["item_id"])][str(row["condition"])] = row
    output.mkdir(parents=True)

    questions = []
    source_records = []
    for item_id in sorted(grouped):
        triplet = grouped[item_id]
        if set(triplet) != set(CONDITIONS):
            raise ValueError(f"{item_id}: incomplete source triplet")
        source = triplet["source_absent"]
        true_row = triplet["record_true"]
        false_row = triplet["record_false"]
        suffix = Path(str(source["source_path"])).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png"}:
            suffix = ".jpg"
        image_name = f"{safe_slug(item_id)}{suffix}"
        destination = output / "images" / image_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source["source_path"], destination)
        if file_sha256(destination) != str(source["source_sha256"]):
            raise AssertionError(f"{item_id}: staged source hash mismatch")
        questions.append({
            "question_id": item_id,
            "image": image_name,
            "text": source["question"],
            "answer": false_row["correct_semantic"],
            "category": "ContraLedger matched three-state delivery",
            "target_text": false_row["registered_read_text"],
            "true_text": true_row["registered_read_text"],
            "target_answer": false_row["target_answer"],
            "source_manifest_sha256": file_sha256(source_manifest),
            "source_image_sha256": source["source_sha256"],
        })
        source_records.append({
            "item_id": item_id,
            "image": image_name,
            "source": source,
            "record_true": true_row,
            "record_false": false_row,
        })

    questions_path = output / "questions.json"
    questions_path.write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    source_rows_path = output / "source_triplets.jsonl"
    source_rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in source_records),
        encoding="utf-8",
    )
    provenance = {
        "schema_version": "cta/contraledger-scenetap-stage-v1",
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "items": len(grouped),
        "questions": str(questions_path),
        "questions_sha256": file_sha256(questions_path),
        "source_triplets": str(source_rows_path),
        "source_triplets_sha256": file_sha256(source_rows_path),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": file_sha256(source_manifest),
        "source_manifest_audit": manifest_audit,
        "family_counts": dict(sorted(Counter(
            triplet["record_false"]["family"] for triplet in grouped.values()
        ).items())),
        "planning_policy": (
            "plan once from the frozen false-record text; reuse the selected region for "
            "the corrected twin; select TextDiffuser candidate zero"
        ),
        "victim_outputs_used": False,
        "official_equivalence": False,
        "claim_boundary": (
            "The downstream replay uses public SceneTAP SoM and TextDiffuser components "
            "with a local Qwen planner, not the unavailable official GPT-4o service."
        ),
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
