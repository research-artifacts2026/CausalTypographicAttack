#!/usr/bin/env python3
"""Attach scene-grounded paired questions to an existing SCEI image manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import file_sha256
from cta.scene_question_designer import QUESTION_VERSION, build_scene_question
from cta.scei_attack import CounterfactualRecord, validate_record


PAIRED_VARIANTS = {"attack_false": "false", "control_true": "true"}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_question_rows(source_rows: list[dict]) -> list[dict]:
    output = []
    by_item: dict[str, list[dict]] = defaultdict(list)
    for row in source_rows:
        variant = str(row.get("variant", ""))
        if variant not in PAIRED_VARIANTS:
            continue
        truth = PAIRED_VARIANTS[variant]
        item_id = str(row["item_id"])
        record = CounterfactualRecord(**row["record"])
        validate_record(record)
        spec = build_scene_question(
            record,
            visible_object=str(row["target_label"]),
            truth=truth,
            item_id=item_id,
        )
        enriched = {
            **row,
            "question_id": f"{item_id}:{variant}",
            "paired_question_id": item_id,
            "scene_question": spec.to_dict(),
            "question": spec.question,
            "options": spec.options,
            "correct_answer": spec.correct_answer,
            "correct_semantic": spec.correct_semantic,
            "registered_attack_target": spec.attack_target_answer,
            "question_generation_uses_victim_outputs": False,
        }
        output.append(enriched)
        by_item[item_id].append(enriched)
    for item_id, rows in by_item.items():
        if {row["variant"] for row in rows} != set(PAIRED_VARIANTS):
            raise ValueError(f"{item_id}: missing false/corrected question twin")
        if len({row["question"] for row in rows}) != 1:
            raise ValueError(f"{item_id}: question differs between semantic twins")
        if {row["correct_semantic"] for row in rows} != {"yes", "no"}:
            raise ValueError(f"{item_id}: paired answers are not counterbalanced")
        if len({json.dumps(row["record"], sort_keys=True) for row in rows}) != 1:
            raise ValueError(f"{item_id}: symbolic record differs between rendered twins")
        if len({json.dumps(row.get("carrier_quad"), sort_keys=True) for row in rows}) != 1:
            raise ValueError(f"{item_id}: carrier geometry differs between rendered twins")
        if len({str(row.get("mask_sha256")) for row in rows}) != 1:
            raise ValueError(f"{item_id}: carrier mask differs between rendered twins")
    return sorted(output, key=lambda row: (str(row["item_id"]), str(row["variant"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    rows = build_question_rows(read_jsonl(source))
    write_jsonl(output, rows)
    provenance = {
        "schema_version": "cta/scei-scene-question-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question_version": QUESTION_VERSION,
        "input": str(source),
        "input_sha256": file_sha256(source),
        "output": str(output),
        "output_sha256": file_sha256(output),
        "items": len({row["item_id"] for row in rows}),
        "rows": len(rows),
        "family_counts": dict(sorted(Counter(row["family"] for row in rows).items())),
        "false_correct_answer_counts": dict(sorted(Counter(
            row["correct_answer"] for row in rows if row["variant"] == "attack_false"
        ).items())),
        "false_true_question_identity": True,
        "victim_outputs_used": False,
        "builder_sha256": file_sha256(Path(__file__).resolve()),
        "question_designer_sha256": file_sha256(
            Path(__file__).resolve().parents[1] / "cta" / "scene_question_designer.py"
        ),
    }
    provenance_path = output.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
