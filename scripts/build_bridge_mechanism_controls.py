#!/usr/bin/env python3
"""Build the frozen Experiment-A Causal-Bridge mechanism manifest.

The input is the ``items.json`` produced by the existing balanced-v1 builder.
No item selection, victim query, or adaptive text search occurs here.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.bridge_mechanism_controls import (
    ALL_CONDITIONS,
    MAX_BODY_WORD_COUNT_SPREAD,
    MECHANISM_CONDITIONS,
    SCHEMA_VERSION,
    read_prompt_for,
    render_condition,
    spec_from_balanced_item,
    validate_manifest_rows,
    validate_text_bundle,
)
from cta.question_bench import file_sha256


def condition_factor_record(condition: str) -> dict:
    factors = {
        "no_attack": (False, "none"),
        "plain": (True, "plain_status"),
        "target_only": (False, "target_aligned"),
        "neutral_only": (False, "neutral"),
        "bridge_aligned": (True, "target_aligned"),
        "bridge_neutral": (True, "neutral"),
        "bridge_reversed": (True, "gold_aligned"),
    }
    proposition_present, conclusion_role = factors[condition]
    return {
        "proposition_present": proposition_present,
        "conclusion_role": conclusion_role,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--balanced-items", type=Path, required=True,
        help="Frozen balanced-v1 items.json; no rows are selected or regenerated.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-items", type=int, required=True)
    parser.add_argument(
        "--stage", choices=("development", "held-out", "transfer"), required=True,
    )
    args = parser.parse_args()

    source_registry = args.balanced_items.resolve()
    loaded = json.loads(source_registry.read_text(encoding="utf-8"))
    if not isinstance(loaded, list) or not loaded:
        raise ValueError("balanced items must be a non-empty JSON list")
    if len(loaded) != args.expected_items:
        raise ValueError(
            f"expected {args.expected_items} frozen items, found {len(loaded)}; "
            "subsampling is not permitted by this builder"
        )
    ids = [str(row.get("item_id", "")) for row in loaded]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("balanced registry has missing or duplicate item IDs")

    specs = [spec_from_balanced_item(row) for row in loaded]
    if {spec.dataset.lower() for spec in specs} != {args.dataset.lower()}:
        raise ValueError("--dataset does not match the frozen item registry")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    text_audits = {}
    for spec in sorted(specs, key=lambda value: value.item_id):
        text_audits[spec.item_id] = validate_text_bundle(spec)
        for condition in ALL_CONDITIONS:
            output = output_root / "images" / condition / f"{spec.item_id}.jpg"
            rendered = render_condition(spec, condition, output)
            row = {
                **spec.to_dict(),
                **rendered,
                **condition_factor_record(condition),
                "answers": [spec.correct_answer_token],
                "target_aliases": [spec.target_answer_token, spec.target_semantic],
                "read_prompt": read_prompt_for(condition),
                "stage": args.stage,
                "protocol_schema_version": SCHEMA_VERSION,
            }
            rows.append(row)

    audit = validate_manifest_rows(rows, check_files=True)
    manifest_path = output_root / "render_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    module_path = Path(__file__).resolve().parents[1] / "cta" / "bridge_mechanism_controls.py"
    source_counts = Counter(spec.counterbalance_cell for spec in specs)
    word_spreads = [audit_row["word_count_spread"] for audit_row in text_audits.values()]
    provenance = {
        "schema_version": f"{SCHEMA_VERSION}/build-provenance",
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "dataset": args.dataset,
        "items": len(specs),
        "rows": len(rows),
        "conditions": list(ALL_CONDITIONS),
        "mechanism_conditions": list(MECHANISM_CONDITIONS),
        "source_registry": str(source_registry),
        "source_registry_sha256": file_sha256(source_registry),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "builder_sha256": file_sha256(Path(__file__)),
        "mechanism_module_sha256": file_sha256(module_path),
        "counterbalance_cell_counts": dict(sorted(source_counts.items())),
        "manifest_audit": audit,
        "text_match_policy": {
            "max_body_word_count_spread": MAX_BODY_WORD_COUNT_SPREAD,
            "observed_max_spread": max(word_spreads),
            "same_title_style_bbox_placement_and_fixed_line_slots": True,
            "explicit_answer_field_or_option_letter": False,
        },
        "question_policy": (
            "verification_question copied byte-for-byte from frozen balanced-v1 item "
            "and held fixed across no_attack plus all six mechanism conditions"
        ),
        "selection_policy": (
            "all frozen registry items included; no victim output, adaptive search, "
            "subsampling, or post-result deletion"
        ),
        "freeze_policy": (
            "runner must receive this exact manifest SHA-256 explicitly in its config"
        ),
        "primary_endpoint": "pooled clean-conditioned read-gated target ASR",
        "primary_interaction": (
            "(bridge_aligned - bridge_neutral) - (target_only - neutral_only)"
        ),
    }
    provenance_path = output_root / "build_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    freeze = {
        "schema_version": f"{SCHEMA_VERSION}/freeze-record",
        "status": "frozen_before_victim_inference",
        "manifest_sha256": provenance["manifest_sha256"],
        "source_registry_sha256": provenance["source_registry_sha256"],
        "builder_sha256": provenance["builder_sha256"],
        "mechanism_module_sha256": provenance["mechanism_module_sha256"],
    }
    (output_root / "freeze_record.json").write_text(
        json.dumps(freeze, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
