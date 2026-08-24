#!/usr/bin/env python3
"""Re-score saved Obj-MC outputs through the official RIO evaluator code."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--rio-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.rio_repo.resolve()
    sys.path.insert(0, str(repo))
    from src.eval_utils.obj_multiple_choices import evaluate_multiple_choice

    rows = read_jsonl(args.predictions.resolve())
    conditions = sorted({str(row["condition"]) for row in rows})
    qids_by_condition = {
        condition: {str(row["question_id"]) for row in rows if row["condition"] == condition}
        for condition in conditions
    }
    if len({frozenset(ids) for ids in qids_by_condition.values()}) != 1:
        raise ValueError("conditions do not contain identical question-id sets")
    official = {}
    per_item = {}
    for condition in conditions:
        items = sorted(
            (row for row in rows if row["condition"] == condition),
            key=lambda row: str(row["question_id"]),
        )
        data = {
            "question_id": [row["question_id"] for row in items],
            "image_id": [row.get("image_id") for row in items],
            "question": [row["question"] for row in items],
            "choices": [row["choices"] for row in items],
            "answer": [row["answers"][0] for row in items],
        }
        scored = evaluate_multiple_choice(
            [row["question"] for row in items],
            [row["prediction"] for row in items],
            data, allow_text_match=True, gt_key="answer",
        )
        official[condition] = {
            "accuracy": scored["accuracy"],
            "n": len(scored["records"]),
        }
        per_item[condition] = {
            str(record["question_id"]): int(record["is_correct"])
            for record in scored["records"]
        }
    eligible = {qid for qid, correct in per_item["no_attack"].items() if correct}
    for condition in conditions:
        failures = sum(not per_item[condition][qid] for qid in eligible)
        official[condition]["n_clean_correct"] = len(eligible)
        official[condition]["clean_conditioned_asr"] = failures / len(eligible) if eligible else None
    result = {
        "schema_version": "cta/rio-official-score-v1",
        "official_code_path": str(repo),
        "official_code_commit": __import__("subprocess").check_output(
            ["git", "-c", f"safe.directory={repo}", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "predictions": str(args.predictions.resolve()),
        "task": "obj_mc",
        "conditions": official,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

