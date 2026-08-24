#!/usr/bin/env python3
"""Build deterministic, paired typographic attacks for public VQA records."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import CONDITIONS, build_spec, file_sha256, render_condition


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--strict", action="store_true", help="Fail instead of recording unsupported rows")
    args = parser.parse_args()

    question_path = args.question_file.resolve()
    records = json.loads(question_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("question file must contain a JSON list")
    ordered = sorted(
        records,
        key=lambda row: hashlib.sha256(
            f"{args.seed}:{row.get('question_id', row.get('id', ''))}".encode()
        ).hexdigest(),
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "render_manifest.jsonl"
    if manifest.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {manifest}")
    rejected: list[dict] = []
    accepted = 0
    for record in ordered:
        if accepted >= args.limit:
            break
        question_id = str(record.get("question_id", record.get("id", "missing")))
        try:
            spec = build_spec(record, args.image_root, args.seed)
        except (ValueError, FileNotFoundError) as exc:
            rejected.append({"question_id": question_id, "reason": str(exc)})
            if args.strict:
                raise
            continue
        for condition in CONDITIONS:
            suffix = Path(spec.image_name).suffix or ".jpg"
            output = output_root / "images" / condition / f"{spec.question_id}{suffix}"
            rendered = render_condition(spec, condition, output)
            append_jsonl(manifest, {**spec.to_dict(), **rendered, "dataset": args.dataset, "seed": args.seed})
        accepted += 1
    if accepted == 0:
        raise RuntimeError("no supported, materialized records were found")
    rejection_path = output_root / "rejected_records.json"
    rejection_path.write_text(json.dumps(rejected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": "cta/question-benchmark-build-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "seed": args.seed,
        "limit": args.limit,
        "accepted_questions": accepted,
        "rendered_rows": accepted * len(CONDITIONS),
        "rejected_questions": len(rejected),
        "question_file": str(question_path),
        "question_file_sha256": file_sha256(question_path),
        "manifest_sha256": file_sha256(manifest),
        "conditions": list(CONDITIONS),
        "target_selection": "explicit/provided distractor first; deterministic type-specific fallback; never model-response-derived",
        "reporting_boundary": "scene_coherent is an in-house plaque baseline, not the public SceneTAP implementation",
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
