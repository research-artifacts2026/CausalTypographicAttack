#!/usr/bin/env python3
"""Independently audit a completed SCEI image-first dataset."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import file_sha256
from cta.scei_attack import CounterfactualRecord, REQUESTED_COUNTERFACTUAL_FAMILIES, validate_record


IMPLEMENTATION_FILES = (
    "cta/scei_attack.py",
    "cta/scei_batch.py",
    "cta/scei_reasoning_families.py",
    "scripts/build_scei_image_dataset.py",
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _measurement_parts(row: dict[str, Any], truth: str) -> list[str]:
    text = row["record"][f"{truth}_measurement"]
    return [part.strip() for part in text.split("|")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    manifest_path = root / "manifest.jsonl"
    selection_path = root / "selection.jsonl"
    provenance_path = root / "provenance.json"
    manifest = _rows(manifest_path)
    selection = _rows(selection_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: list[str] = []
    image_hash_cache: dict[Path, str] = {}
    mask_hash_cache: dict[Path, str] = {}

    for row in manifest:
        by_item[str(row["item_id"])].append(row)
        image_path = root / row["image_relative_path"]
        if not image_path.is_file():
            errors.append(f"missing image: {image_path}")
            continue
        actual = image_hash_cache.setdefault(image_path, file_sha256(image_path))
        if actual != row["image_sha256"]:
            errors.append(f"image hash mismatch: {image_path}")
        if row.get("mask_relative_path"):
            mask_path = root / row["mask_relative_path"]
            if not mask_path.is_file():
                errors.append(f"missing mask: {mask_path}")
            else:
                mask_actual = mask_hash_cache.setdefault(mask_path, file_sha256(mask_path))
                if mask_actual != row["mask_sha256"]:
                    errors.append(f"mask hash mismatch: {mask_path}")
        if float(row["overlay_area_fraction"]) > float(provenance["max_area_fraction"]) + 1e-6:
            errors.append(f"area cap exceeded: {row['item_id']}:{row['variant']}")

    variants_expected = {"clean", "attack_false", "control_true"}
    one_field_pairs = 0
    geometry_matched = 0
    mask_matched = 0
    clean_source_matched = 0
    distinct_false_true = 0
    validated_symbolic_records = 0
    semantic_signatures: set[str] = set()
    duplicate_semantic_signatures: list[str] = []
    scene_conditioned_records = 0
    for item_id, rows in by_item.items():
        variants = {str(row["variant"]): row for row in rows}
        if set(variants) != variants_expected or len(rows) != 3:
            errors.append(f"{item_id}: incomplete or duplicate variants")
            continue
        clean, false, true = (variants[name] for name in ("clean", "attack_false", "control_true"))
        if clean["image_sha256"] == clean["source_sha256"]:
            clean_source_matched += 1
        else:
            errors.append(f"{item_id}: clean copy differs from source hash")
        if false["carrier_quad"] == true["carrier_quad"]:
            geometry_matched += 1
        else:
            errors.append(f"{item_id}: carrier geometry differs")
        if false["mask_sha256"] == true["mask_sha256"]:
            mask_matched += 1
        else:
            errors.append(f"{item_id}: false/true masks differ")
        false_parts = _measurement_parts(false, "false")
        true_parts = _measurement_parts(true, "true")
        if len(false_parts) == len(true_parts) and sum(a != b for a, b in zip(false_parts, true_parts)) == 1:
            one_field_pairs += 1
        else:
            errors.append(f"{item_id}: false/true records do not differ in exactly one field")
        if false["image_sha256"] != true["image_sha256"]:
            distinct_false_true += 1
        else:
            errors.append(f"{item_id}: false/true image hashes are identical")
        try:
            symbolic_record = CounterfactualRecord(**false["record"])
            validate_record(symbolic_record)
            validated_symbolic_records += 1
            if symbolic_record.generator_version == "scei-symbolic-v2":
                anchor = str(symbolic_record.parameters.get("scene_anchor_label", "")).strip().lower()
                target = str(false.get("target_label", "")).strip().lower()
                expected_tag = re.sub(r"[^A-Z0-9]+", " ", target.upper()).strip()[:22].rstrip()
                if anchor != target:
                    errors.append(f"{item_id}: symbolic scene anchor does not match target label")
                elif not symbolic_record.false_measurement.startswith(expected_tag):
                    errors.append(f"{item_id}: printed record does not name the visible anchor")
                elif false.get("content_conditioning", {}).get("victim_outputs_used") is not False:
                    errors.append(f"{item_id}: content-conditioning provenance is missing the victim-output boundary")
                else:
                    scene_conditioned_records += 1
        except Exception as exc:
            errors.append(f"{item_id}: symbolic validation failed: {exc}")
        if false.get("split") != true.get("split") or false.get("split") != clean.get("split"):
            errors.append(f"{item_id}: variants cross dataset splits")
        signature = str(false.get("semantic_signature", "")).strip()
        if signature:
            if signature in semantic_signatures:
                duplicate_semantic_signatures.append(signature)
                errors.append(f"{item_id}: duplicate semantic signature")
            semantic_signatures.add(signature)

    family_counts = dict(sorted(Counter(rows[0]["family"] for rows in by_item.values()).items()))
    split_counts = dict(sorted(Counter(str(rows[0].get("split", "unspecified")) for rows in by_item.values()).items()))
    if provenance.get("record_generator") == "diverse_v2":
        missing_families = set(REQUESTED_COUNTERFACTUAL_FAMILIES) - set(family_counts)
        if missing_families:
            errors.append(f"missing registered families: {sorted(missing_families)}")
        if len(set(family_counts.values())) != 1:
            errors.append(f"v2 family counts are not balanced: {family_counts}")
        if len(semantic_signatures) != len(by_item):
            errors.append("v2 semantic signatures are not unique per item")
        if family_counts != provenance.get("family_counts"):
            errors.append("family counts disagree with provenance")
        if split_counts != provenance.get("split_item_counts"):
            errors.append("split counts disagree with provenance")

    report = {
        "schema_version": "cta/scei-image-audit-v1",
        "status": "pass" if not errors else "fail",
        "dataset_root": str(root),
        "selection_rows": len(selection),
        "manifest_rows": len(manifest),
        "items": len(by_item),
        "unique_image_files": len(image_hash_cache),
        "unique_mask_files": len(mask_hash_cache),
        "variant_counts": dict(sorted(Counter(str(row["variant"]) for row in manifest).items())),
        "family_item_counts": family_counts,
        "split_item_counts": split_counts,
        "difficulty_item_counts": dict(sorted(Counter(
            str(rows[0]["record"].get("difficulty", "canonical")) for rows in by_item.values()
        ).items())),
        "scenario_item_counts": dict(sorted(Counter(str(rows[0]["scenario_id"]) for rows in by_item.values()).items())),
        "unique_semantic_signatures": len(semantic_signatures),
        "duplicate_semantic_signatures": len(duplicate_semantic_signatures),
        "validated_symbolic_records": validated_symbolic_records,
        "scene_conditioned_records": scene_conditioned_records,
        "planner_valid_items": sum(bool(rows[0]["planner_valid"]) for rows in by_item.values()),
        "clean_source_hash_matches": clean_source_matched,
        "one_field_counterfactual_pairs": one_field_pairs,
        "geometry_matched_pairs": geometry_matched,
        "mask_matched_pairs": mask_matched,
        "distinct_false_true_images": distinct_false_true,
        "manifest_sha256_matches_provenance": file_sha256(manifest_path) == provenance["manifest_sha256"],
        "selection_sha256_matches_provenance": file_sha256(selection_path) == provenance["selection_sha256"],
        "implementation_hashes_match": all(
            (Path(__file__).resolve().parents[1] / relative).is_file()
            and file_sha256(Path(__file__).resolve().parents[1] / relative) == expected
            for relative, expected in provenance.get("implementation_files_sha256", {}).items()
        ) if provenance.get("implementation_files_sha256") else None,
        "errors": errors[:100],
        "error_count": len(errors),
    }
    output = root / "audit_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
