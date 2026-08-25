#!/usr/bin/env python3
"""Freeze one universal CTA-v2 template from development logs only."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import TARGET_AWARE_CANDIDATES, file_sha256


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-log", action="append", required=True, help="MODEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-option-anchor", action="store_true",
        help="Allow the option-letter upper bound to become the selected primary template",
    )
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    manifest_rows = read_jsonl(manifest)
    candidates = sorted(
        {row["condition"] for row in manifest_rows} & set(TARGET_AWARE_CANDIDATES)
    )
    if not candidates:
        raise ValueError("development manifest has no CTA-v2 candidates")
    evidence: dict[str, dict] = {}
    aggregate = {condition: [] for condition in candidates}
    for assignment in args.model_log:
        model, value = assignment.split("=", 1)
        path = Path(value).resolve()
        rows = read_jsonl(path)
        by_condition = {
            condition: {row["question_id"]: row for row in rows if row["condition"] == condition}
            for condition in ("no_attack", *candidates)
        }
        clean_correct = {
            qid for qid, row in by_condition["no_attack"].items()
            if float(row["answer_score"]) >= 1.0
        }
        if not clean_correct:
            raise ValueError(f"{model}: no clean-correct development questions")
        scores = {}
        for condition in candidates:
            if set(by_condition[condition]) != set(by_condition["no_attack"]):
                raise ValueError(f"{model}: incomplete candidate {condition}")
            targeted = sum(
                bool(by_condition[condition][qid]["target_match"]) for qid in clean_correct
            ) / len(clean_correct)
            untargeted = sum(
                float(by_condition[condition][qid]["answer_score"]) < 1.0
                for qid in clean_correct
            ) / len(clean_correct)
            scores[condition] = {
                "n_clean_correct": len(clean_correct),
                "targeted_asr": targeted,
                "clean_conditioned_asr": untargeted,
            }
            aggregate[condition].append((targeted, untargeted))
        evidence[model] = {"log": str(path), "log_sha256": file_sha256(path), "scores": scores}

    macro = {
        condition: {
            "macro_targeted_asr": sum(value[0] for value in values) / len(values),
            "macro_clean_conditioned_asr": sum(value[1] for value in values) / len(values),
        }
        for condition, values in aggregate.items()
    }
    selection_pool = [
        condition for condition in candidates
        if args.include_option_anchor or condition != "cta_option_anchor"
    ]
    selected = max(
        selection_pool,
        key=lambda condition: (
            macro[condition]["macro_targeted_asr"],
            macro[condition]["macro_clean_conditioned_asr"],
            condition,
        ),
    )
    result = {
        "schema_version": "cta/template-preregistration-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-held-out-rendering",
        "selection_rule": (
            "maximum macro targeted ASR across registered development models; "
            "macro clean-conditioned ASR then lexical condition name break ties"
        ),
        "development_manifest": str(manifest),
        "development_manifest_sha256": file_sha256(manifest),
        "candidates": candidates,
        "selection_pool": selection_pool,
        "models": evidence,
        "aggregate": macro,
        "selected_condition": selected,
        "held_out_results_seen": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite preregistration: {args.output}")
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
