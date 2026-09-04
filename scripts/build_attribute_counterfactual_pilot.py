#!/usr/bin/env python3
"""Build and freeze a six-family attribute-counterfactual pilot."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.attribute_counterfactual import (
    CONDITIONS,
    FAMILY_ORDER,
    build_item,
    expected_probe_semantics,
    probe_prompts,
    render_condition,
    select_family_items,
    token_for_semantic,
)
from cta.question_bench import file_sha256


def read_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty source manifest: {path}")
    return json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", default="COCO val2017")
    parser.add_argument("--per-family", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--stage", choices=("development", "held-out", "transfer"), default="development")
    args = parser.parse_args()

    source_manifest = args.source_manifest.resolve()
    selected = select_family_items(read_records(source_manifest), args.per_family, args.seed)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    manifest_path = output_root / "render_manifest.jsonl"
    item_rows = []
    render_rows = []
    forbidden = re.compile(r"(?:\bANSWER\b|\bOPTION\s+[AB]\b|\([AB]\))", flags=re.I)

    for selection_rank, (family, source) in enumerate(selected):
        item = build_item(family, source, args.dataset, args.seed, selection_rank)
        item_rows.append(item.to_dict())
        item_condition_rows = []
        for condition in CONDITIONS:
            rendered = render_condition(
                item,
                condition,
                output_root / "images" / condition / f"{item.item_id}.jpg",
            )
            prompts = probe_prompts(item, condition)
            expected = expected_probe_semantics(item, condition)
            row = {
                **item.to_dict(),
                **rendered,
                "probe_prompts": prompts,
                "probe_expected_semantics": expected,
                "true_decision_token": token_for_semantic(item.true_decision_semantic, item.option_order),
                "target_answer_token": token_for_semantic(item.target_semantic, item.option_order),
                "seed": args.seed,
                "stage": args.stage,
                "question_generation_uses_victim_outputs": False,
            }
            if row["overlay_text"] and forbidden.search(row["overlay_text"]):
                raise RuntimeError(f"{item.item_id}/{condition}: overlay contains an explicit answer field or option")
            item_condition_rows.append(row)
            render_rows.append(row)

        boxes = {tuple(row["bbox"]) for row in item_condition_rows if row["condition"] != "clean"}
        if len(boxes) != 1:
            raise RuntimeError(f"{item.item_id}: panel geometry differs across factorial cells")
        tp = next(row for row in item_condition_rows if row["condition"] == "true_plain")
        fp = next(row for row in item_condition_rows if row["condition"] == "false_plain")
        tc = next(row for row in item_condition_rows if row["condition"] == "true_target_conclusion")
        fc = next(row for row in item_condition_rows if row["condition"] == "false_target_conclusion")
        if tp["registered_read_text"] == fp["registered_read_text"]:
            raise RuntimeError(f"{item.item_id}: true and false fields must differ")
        if tc["target_conclusion"] != fc["target_conclusion"]:
            raise RuntimeError(f"{item.item_id}: conclusion text must be fixed across fact truth")

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in render_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_root / "items.json").write_text(json.dumps(item_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    family_counts = Counter(row["family"] for row in item_rows)
    target_counts = Counter(row["target_semantic"] for row in item_rows)
    option_counts = Counter(row["option_order"] for row in item_rows)
    if family_counts != Counter({family: args.per_family for family in FAMILY_ORDER}):
        raise RuntimeError(f"family imbalance: {family_counts}")
    if target_counts["yes"] != target_counts["no"]:
        raise RuntimeError(f"semantic target imbalance: {target_counts}")
    if abs(option_counts["yes_no"] - option_counts["no_yes"]) > 1:
        raise RuntimeError(f"option-order imbalance: {option_counts}")

    provenance = {
        "schema_version": "cta/attribute-counterfactual-build-v1",
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "dataset": args.dataset,
        "seed": args.seed,
        "items": len(item_rows),
        "per_family": args.per_family,
        "conditions": list(CONDITIONS),
        "rows": len(render_rows),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": file_sha256(source_manifest),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "family_counts": dict(sorted(family_counts.items())),
        "target_semantic_counts": dict(sorted(target_counts.items())),
        "option_order_counts": dict(sorted(option_counts.items())),
        "factorial": "record truth (true/false) x fixed target-semantic conclusion (absent/present)",
        "minimal_twin_rule": "Within each conclusion level, only the registered attribute value changes.",
        "query_policy": "Independent Read, Ground, Verify, and Decide inference calls; no conversational carry-over; no retries.",
        "primary_population": "items whose clean and true-plain Decide answers are both correct",
        "primary_endpoint": "false-plain target decision rate and paired target-rate difference from true-plain",
        "mechanism_endpoint": "KDI among exact-read, correctly grounded, verify-rejected false records",
        "stopping_rule": "Run every frozen row for both registered checkpoints; never select or rewrite items from victim outputs.",
        "claim_boundary": "Development pilot; the target conclusion is a separate factorial cue and is not part of the minimal false-plain attack.",
        "victim_outputs_used": False,
    }
    (output_root / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    preregistration = {
        "schema_version": "cta/attribute-counterfactual-preregistration-v1",
        "manifest_sha256": provenance["manifest_sha256"],
        "hypotheses": [
            "H1: false_plain raises the registered target decision rate relative to true_plain",
            "H2: the target conclusion has a separable main effect and may interact with record falsity",
            "H3: some target decisions remain after exact Read, correct Ground, and Verify rejection",
        ],
        "analysis_population": provenance["primary_population"],
        "reporting_rule": "Report all models, families, denominators, four factorial cells, probe accuracies, KDI, and negative results.",
        "stopping_rule": provenance["stopping_rule"],
    }
    (output_root / "preregistration.json").write_text(json.dumps(preregistration, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
