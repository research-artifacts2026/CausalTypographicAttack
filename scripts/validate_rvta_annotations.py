#!/usr/bin/env python3
"""Validate blinded RVTA JSONL annotations and enforce independent coverage."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.rvta_bench import validate_annotation_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("responses", type=Path)
    parser.add_argument("--minimum-annotators", type=int, default=3)
    args = parser.parse_args()
    if args.minimum_annotators < 1:
        raise ValueError("minimum annotators must be positive")

    rows = []
    for line_number, line in enumerate(args.responses.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            validate_annotation_record(row)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid annotation at line {line_number}: {exc}") from exc
        rows.append(row)
    if not rows:
        raise ValueError("annotation file is empty")

    keys = [(row["item_id"], row["annotator_id"]) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate item/annotator rows: {len(duplicates)}")
    coverage = defaultdict(set)
    for row in rows:
        coverage[row["item_id"]].add(row["annotator_id"])
    undercovered = {
        item_id: len(annotators)
        for item_id, annotators in coverage.items()
        if len(annotators) < args.minimum_annotators
    }
    if undercovered:
        raise ValueError(
            f"{len(undercovered)} items have fewer than {args.minimum_annotators} independent annotators"
        )
    print(json.dumps({
        "rows": len(rows),
        "items": len(coverage),
        "annotators": len({row["annotator_id"] for row in rows}),
        "minimum_annotators": min(len(value) for value in coverage.values()),
    }))


if __name__ == "__main__":
    main()
