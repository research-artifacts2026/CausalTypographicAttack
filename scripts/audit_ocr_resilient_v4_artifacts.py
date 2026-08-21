#!/usr/bin/env python3
"""Independently audit frozen OCR-resilient v4 render artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _outside_bbox_equal(clean_path: Path, image_path: Path, bbox: list[int]) -> bool:
    clean = np.asarray(Image.open(clean_path).convert("RGB"))
    image = np.asarray(Image.open(image_path).convert("RGB"))
    if clean.shape != image.shape:
        return False
    x0, y0, x1, y1 = bbox
    outside = np.ones(clean.shape[:2], dtype=bool)
    outside[y0:y1, x0:x1] = False
    return bool(np.array_equal(clean[outside], image[outside]))


def audit_split(run_root: Path, reserved: set[str]) -> tuple[dict, set[str]]:
    conditions = _jsonl(run_root / "conditions.jsonl")
    candidates = _jsonl(run_root / "candidate_search.jsonl")
    sample_ids = {row["sample_id"] for row in conditions}
    candidate_counts = Counter(row["sample_id"] for row in candidates)

    hashes_ok = True
    pixel_locality_ok = True
    for row in conditions:
        attack = row["attack_metadata"]
        defense = row["defense_metadata"]
        hashes_ok &= _sha256(Path(attack["image_path"])) == attack["rendered_sha256"]
        hashes_ok &= _sha256(Path(defense["image_path"])) == defense["defended_sha256"]
        hashes_ok &= _sha256(Path(attack["carrier_mask_path"])) == attack["carrier_mask_sha256"]
        clean_path = Path(row["clean_image_path"])
        bbox = attack["layout_bbox"]
        pixel_locality_ok &= _outside_bbox_equal(clean_path, Path(attack["image_path"]), bbox)
        pixel_locality_ok &= _outside_bbox_equal(clean_path, Path(defense["image_path"]), bbox)

    survival = [row["defense_metadata"]["carrier_survival_fraction"] for row in conditions]
    min_delta_e = [
        row["attack_metadata"]["selected_legibility"]["minimum_word_delta_e76"]
        for row in conditions
    ]
    summary = {
        "run_root": str(run_root),
        "rows": len(conditions),
        "unique_ids": len(sample_ids),
        "reserved_overlap": len(sample_ids & reserved),
        "readability_gate_pass": sum(
            bool(row["attack_metadata"]["readability_gate_passed"]) for row in conditions
        ),
        "candidate_rows": len(candidates),
        "candidate_counts": sorted(set(candidate_counts.values())),
        "candidate_outside_nonzero": sum(
            row["changed_pixels_outside_layout_bbox_relative_to_clean"] != 0
            for row in candidates
        ),
        "selected_outside_nonzero": sum(
            row["attack_metadata"]["changed_pixels_outside_layout_bbox"] != 0
            or row["defense_metadata"]["changed_pixels_outside_layout_bbox_relative_to_clean"] != 0
            for row in conditions
        ),
        "hashes_ok": bool(hashes_ok),
        "independent_pixel_locality_ok": bool(pixel_locality_ok),
        "max_layout": max(row["attack_metadata"]["layout_area_fraction"] for row in conditions),
        "max_occlusion": max(
            row["attack_metadata"]["object_bbox_occlusion_fraction"] for row in conditions
        ),
        "min_survival": min(survival),
        "mean_survival": sum(survival) / len(survival),
        "min_word_delta_e_min": min(min_delta_e),
    }
    return summary, sample_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--discovery-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument(
        "--additional-root",
        type=Path,
        action="append",
        default=[],
        help="Optional extra immutable condition root, audited without changing discovery/test semantics.",
    )
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text())
    reserved = set(registry["reserved_ids"])
    discovery, discovery_ids = audit_split(args.discovery_root, reserved)
    test, test_ids = audit_split(args.test_root, reserved)
    result = {
        "registry_count": len(reserved),
        "registry_hash": registry["canonical_ids_sha256"],
        "discovery_test_overlap": len(discovery_ids & test_ids),
        "discovery": discovery,
        "test": test,
        "additional": {},
    }
    for root in args.additional_root:
        summary, ids = audit_split(root, reserved)
        result["additional"][str(root)] = {
            **summary,
            "overlap_with_discovery": len(ids & discovery_ids),
            "overlap_with_test": len(ids & test_ids),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
