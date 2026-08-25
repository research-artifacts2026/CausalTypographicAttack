#!/usr/bin/env python3
"""Integrity audit for frozen balanced RVTA-QA manifests."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import file_sha256
from cta.rvta_qa_balanced import CONDITIONS


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(path: Path) -> dict:
    rows = read_jsonl(path)
    keys = [(row["item_id"], row["condition"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate item-condition keys")
    if {row["condition"] for row in rows} != set(CONDITIONS):
        raise ValueError(f"{path}: condition set mismatch")
    by_item: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_item[row["item_id"]].append(row)
    image_hashes_checked = 0
    source_hashes_checked = 0
    cell_counts = Counter()
    attack_areas = []
    item_area_deltas = []
    for item_id, item_rows in by_item.items():
        if len(item_rows) != len(CONDITIONS) or {row["condition"] for row in item_rows} != set(CONDITIONS):
            raise ValueError(f"{path}: incomplete condition set for {item_id}")
        fixed_fields = (
            "verification_question", "verification_claim", "proposition_truth", "answer_format",
            "option_order", "counterbalance_cell", "correct_semantic", "target_semantic",
            "source_path", "source_sha256",
        )
        for field in fixed_fields:
            if len({json.dumps(row[field], sort_keys=True) for row in item_rows}) != 1:
                raise ValueError(f"{path}: non-invariant {field} for {item_id}")
        cell_counts[item_rows[0]["counterbalance_cell"]] += 1
        source = Path(item_rows[0]["source_path"])
        if file_sha256(source) != item_rows[0]["source_sha256"]:
            raise ValueError(f"{path}: source hash mismatch for {item_id}")
        source_hashes_checked += 1
        item_areas = []
        for row in item_rows:
            image = Path(row["image_path"])
            if file_sha256(image) != row["image_sha256"]:
                raise ValueError(f"{path}: image hash mismatch for {(item_id, row['condition'])}")
            image_hashes_checked += 1
            if row["condition"] == "no_attack":
                if row["bbox"] is not None or float(row["overlay_area_fraction"]) != 0.0:
                    raise ValueError(f"{path}: clean geometry is not empty for {item_id}")
            else:
                area = float(row["overlay_area_fraction"])
                item_areas.append(area)
                attack_areas.append(area)
        item_area_deltas.append(max(item_areas) - min(item_areas))
    if max(cell_counts.values()) - min(cell_counts.values()) > 1:
        raise ValueError(f"{path}: counterbalance cells differ by more than one")
    return {
        "manifest": str(path),
        "manifest_sha256": file_sha256(path),
        "items": len(by_item),
        "rows": len(rows),
        "conditions": list(CONDITIONS),
        "counterbalance_cell_counts": dict(sorted(cell_counts.items())),
        "source_hashes_checked": source_hashes_checked,
        "image_hashes_checked": image_hashes_checked,
        "question_invariant": True,
        "geometry_identical_items": sum(delta <= 1e-12 for delta in item_area_deltas),
        "geometry_variable_items": sum(delta > 1e-12 for delta in item_area_deltas),
        "mean_within_item_area_delta": statistics.fmean(item_area_deltas),
        "median_within_item_area_delta": statistics.median(item_area_deltas),
        "maximum_within_item_area_delta": max(item_area_deltas),
        "geometry_note": (
            "Text-dependent font fitting can change panel height; deltas are audited and must not be described as exact-area control."
        ),
        "mean_attack_area_fraction": statistics.fmean(attack_areas),
        "minimum_attack_area_fraction": min(attack_areas),
        "maximum_attack_area_fraction": max(attack_areas),
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema_version": "cta/rvta-qa-balanced-frozen-manifest-audit-v1",
        "status": "passed",
        "manifests": [audit(path.resolve()) for path in args.manifest],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
