#!/usr/bin/env python3
"""Freeze a disjoint three-state ContraLedger confirmation set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.contraledger_threeway import CONDITIONS, OPTION_PERMUTATIONS, render_item
from cta.question_bench import file_sha256
from cta.scei_reasoning_families import FAMILY_IDS
from scripts.build_contraledger import paired_sources, select_balanced


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def item_ids(path: Path) -> set[str]:
    return {str(row["item_id"]) for row in read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--per-family", type=int, default=25)
    parser.add_argument("--offset-per-family", type=int, default=75)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--max-area-fraction", type=float, default=0.15)
    args = parser.parse_args()

    source = args.source_manifest.resolve()
    output = args.output_root.resolve()
    selected = select_balanced(
        paired_sources(read_jsonl(source)),
        per_family=args.per_family,
        offset_per_family=args.offset_per_family,
        seed=args.seed,
    )
    selected_ids = {str(row["item_id"]) for row in selected}
    excluded = set()
    excluded_records = []
    for path in args.exclude_manifest:
        resolved = path.resolve()
        ids = item_ids(resolved)
        overlap = selected_ids & ids
        if overlap:
            raise ValueError(f"confirmation split overlaps {resolved}: {len(overlap)} items")
        excluded |= ids
        excluded_records.append({"path": str(resolved), "sha256": file_sha256(resolved), "items": len(ids)})

    output.mkdir(parents=True, exist_ok=False)

    rows = []
    family_index = {family: 0 for family in FAMILY_IDS}
    for row in selected:
        family = str(row["family"])
        index = (family_index[family] + FAMILY_IDS.index(family)) % len(OPTION_PERMUTATIONS)
        family_index[family] += 1
        rows.extend(render_item(
            row,
            output,
            permutation_index=index,
            max_area_fraction=args.max_area_fraction,
        ))
    manifest = output / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    family_counts = Counter(row["family"] for row in selected)
    option_counts = Counter(row["option_permutation_index"] for row in rows if row["condition"] == "source_absent")
    provenance = {
        "schema_version": "cta/contraledger-threeway-build-v1",
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "confirmatory",
        "seed": args.seed,
        "items": len(selected),
        "rows": len(rows),
        "conditions": list(CONDITIONS),
        "family_counts": dict(sorted(family_counts.items())),
        "option_permutation_counts": {str(k): v for k, v in sorted(option_counts.items())},
        "offset_per_family": args.offset_per_family,
        "source_manifest": str(source),
        "source_manifest_sha256": file_sha256(source),
        "excluded_manifests": excluded_records,
        "excluded_item_count": len(excluded),
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "question_policy": (
            "one scene-specific three-choice question, byte-identical across source/true/false; "
            "registered answers are absent/consistent/inconsistent"
        ),
        "selection_policy": (
            "remaining disjoint SHA-256-ordered items from the pre-existing frozen source pool; "
            "no victim outputs used"
        ),
        "victim_outputs_used": False,
        "primary_endpoint": (
            "false-record CONSISTENT target ASR conditioned on correct source-ABSENT and true-record-CONSISTENT controls"
        ),
        "mechanism_endpoint": "EOR additionally requires exact false-record reading and correct independent rejection",
        "claim_boundary": "Controlled digital carrier; no human-naturalness or camera-capture claim.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    preregistration = {
        "schema_version": "cta/contraledger-threeway-preregistration-v1",
        "manifest_sha256": provenance["manifest_sha256"],
        "hypotheses": [
            "H1: values-only false records induce nonzero CONSISTENT target ASR after both valid controls are correct",
            "H2: EOR is nonzero after exact reading and correct independent rule rejection",
        ],
        "reporting_rule": "Report every model, family, denominator, confidence interval, parse failure, and negative result.",
        "stopping_rule": "Run every frozen row once; do not alter questions, records, items, or rendering from victim outputs.",
    }
    (output / "preregistration.json").write_text(json.dumps(preregistration, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
