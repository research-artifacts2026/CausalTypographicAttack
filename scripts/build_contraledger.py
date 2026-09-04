#!/usr/bin/env python3
"""Build a frozen balanced ContraLedger truth-by-cue dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.contraledger import CONDITIONS, CUE_LEVELS, render_factorial_item
from cta.question_bench import file_sha256
from cta.scei_reasoning_families import FAMILY_IDS


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rank(seed: int, item_id: str) -> str:
    return hashlib.sha256(f"contraledger-v1:{seed}:{item_id}".encode()).hexdigest()


def paired_sources(rows: list[dict]) -> list[dict]:
    by_item: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("variant") in {"attack_false", "control_true"}:
            by_item[str(row["item_id"])].append(row)
    result = []
    for item_id, pair in by_item.items():
        if {str(row["variant"]) for row in pair} != {"attack_false", "control_true"}:
            raise ValueError(f"{item_id}: incomplete false/corrected source pair")
        if len({str(row["question"]) for row in pair}) != 1:
            raise ValueError(f"{item_id}: source question differs across twins")
        if len({json.dumps(row["record"], sort_keys=True) for row in pair}) != 1:
            raise ValueError(f"{item_id}: source symbolic record differs across twins")
        result.append(next(row for row in pair if row["variant"] == "attack_false"))
    return result


def select_balanced(
    rows: list[dict], *, per_family: int, offset_per_family: int, seed: int
) -> list[dict]:
    selected = []
    for family in FAMILY_IDS:
        eligible = sorted(
            (row for row in rows if row["family"] == family),
            key=lambda row: _rank(seed, str(row["item_id"])),
        )
        subset = eligible[offset_per_family : offset_per_family + per_family]
        if len(subset) != per_family:
            raise ValueError(
                f"{family}: requested {per_family} after offset {offset_per_family}, found {len(subset)}"
            )
        family_index = FAMILY_IDS.index(family)
        for within_family_index, row in enumerate(subset):
            selected.append({
                **row,
                "_question_polarity": (
                    "positive" if (within_family_index + family_index) % 2 == 0 else "negative"
                ),
            })
    return sorted(selected, key=lambda row: (FAMILY_IDS.index(row["family"]), _rank(seed, row["item_id"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-family", type=int, default=15)
    parser.add_argument("--offset-per-family", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--stage", choices=("development", "held-out", "transfer"), default="development")
    parser.add_argument("--max-area-fraction", type=float, default=0.15)
    args = parser.parse_args()

    source = args.source_manifest.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    selected = select_balanced(
        paired_sources(read_jsonl(source)),
        per_family=args.per_family,
        offset_per_family=args.offset_per_family,
        seed=args.seed,
    )
    rows = []
    for source_row in selected:
        rows.extend(render_factorial_item(
            source_row, output, max_area_fraction=args.max_area_fraction
        ))
    expected = len(selected) * len(CONDITIONS)
    if len(rows) != expected:
        raise RuntimeError(f"incomplete render: {len(rows)}/{expected}")
    manifest = output / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    family_counts = Counter(row["family"] for row in selected)
    polarity_counts = Counter(row["_question_polarity"] for row in selected)
    if abs(polarity_counts["positive"] - polarity_counts["negative"]) > 1:
        raise RuntimeError(f"question-polarity imbalance: {polarity_counts}")
    provenance = {
        "schema_version": "cta/contraledger-build-v1",
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "seed": args.seed,
        "items": len(selected),
        "rows": len(rows),
        "conditions": list(CONDITIONS),
        "cue_levels": list(CUE_LEVELS),
        "family_counts": dict(sorted(family_counts.items())),
        "question_polarity_counts": dict(sorted(polarity_counts.items())),
        "offset_per_family": args.offset_per_family,
        "source_manifest": str(source),
        "source_manifest_sha256": file_sha256(source),
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "question_policy": (
            "one source-derived neutral question, byte-identical across all six item conditions; "
            "positive/negative consistency polarity balanced before victim inference"
        ),
        "factorial": "record truth (true/false) x cue (values-only/authority/explicit conclusion)",
        "renderer": "same deterministic scene carrier and mask across all six conditions",
        "selection_policy": "balanced SHA-256 ordering from the frozen source manifest; no victim outputs",
        "victim_outputs_used": False,
        "primary_endpoint": "same-cue-true-twin-conditioned target ASR for each false cue condition",
        "mechanism_endpoint": "EOR after exact false-record transcription and correct independent knowledge judgment",
        "claim_boundary": (
            "Explicit conclusion is an attack upper bound. Values-only isolates numeric/relational content; "
            "authority is non-evaluative. No naturalness or physical-capture claim follows."
        ),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    preregistration = {
        "schema_version": "cta/contraledger-preregistration-v1",
        "manifest_sha256": provenance["manifest_sha256"],
        "hypotheses": [
            "H1: false values-only records induce nonzero target ASR on same-cue true-twin-correct items",
            "H2: authority increases target ASR over values-only on the common true-twin-correct population",
            "H3: explicit conclusion is an upper bound and increases target ASR over values-only",
            "H4: EOR is nonzero after exact reading and correct independent rule verification",
        ],
        "reporting_rule": "Report all cue levels, models, families, denominators, confidence intervals, and negative results.",
        "stopping_rule": "Run every frozen row exactly once per probe and checkpoint; no item or wording selection from victim outputs.",
    }
    (output / "preregistration.json").write_text(
        json.dumps(preregistration, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
