#!/usr/bin/env python3
"""Create an immutable development or held-out manifest with CTA-v2 cards."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import (
    TARGET_AWARE_CANDIDATES, QuestionAttackSpec, file_sha256, render_condition,
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def spec_from_row(row: dict) -> QuestionAttackSpec:
    names = {field.name for field in fields(QuestionAttackSpec)}
    values = {name: row[name] for name in names}
    values["answers"] = tuple(values["answers"])
    values["target_aliases"] = tuple(values["target_aliases"])
    return QuestionAttackSpec(**values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--include-condition", action="append", default=[])
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--stage", choices=("development", "held-out"), required=True)
    args = parser.parse_args()

    source_manifest = args.source_manifest.resolve()
    source_rows = read_jsonl(source_manifest)
    if not source_rows:
        raise ValueError("source manifest is empty")
    candidates = tuple(args.candidate or TARGET_AWARE_CANDIDATES)
    unknown = sorted(set(candidates) - set(TARGET_AWARE_CANDIDATES))
    if unknown:
        raise ValueError(f"unknown candidates: {unknown}")
    if args.stage == "held-out" and len(candidates) != 1:
        raise ValueError("held-out manifests must freeze exactly one candidate")

    by_key = {(str(row["question_id"]), row["condition"]): row for row in source_rows}
    if len(by_key) != len(source_rows):
        raise ValueError("source manifest has duplicate question-condition keys")
    question_ids = sorted({str(row["question_id"]) for row in source_rows})
    includes = tuple(args.include_condition or ("no_attack",))
    missing = [
        (qid, condition) for qid in question_ids for condition in includes
        if (qid, condition) not in by_key
    ]
    if missing:
        raise ValueError(f"source manifest is missing requested base rows: {missing[:5]}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "render_manifest.jsonl"
    if manifest.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {manifest}")

    rows_written = 0
    for qid in question_ids:
        clean = by_key[(qid, "no_attack")]
        for condition in includes:
            append_jsonl(manifest, by_key[(qid, condition)])
            rows_written += 1
        spec = spec_from_row(clean)
        for condition in candidates:
            output = output_root / "images" / condition / f"{qid}.jpg"
            rendered = render_condition(spec, condition, output)
            append_jsonl(manifest, {
                **clean, **spec.to_dict(), **rendered,
                "template_selection_stage": args.stage,
            })
            rows_written += 1

    provenance = {
        "schema_version": "cta/target-aware-manifest-v1",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": file_sha256(source_manifest),
        "questions": len(question_ids),
        "included_base_conditions": list(includes),
        "candidate_conditions": list(candidates),
        "rows": rows_written,
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "selection_boundary": (
            "All templates may be evaluated only at development stage; held-out stage "
            "requires one preselected template and forbids per-image or per-model selection."
        ),
        "option_anchor_boundary": (
            "cta_option_anchor exposes the target option letter and is an adaptive upper "
            "bound, not the primary reality-violation method."
        ),
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
