#!/usr/bin/env python3
"""Audit a contextual source manifest without evaluating a victim model."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.source_manifest.resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty source manifest")
    item_ids = [row["item_id"] for row in rows]
    fact_ids = [row["fact"]["fact_id"] for row in rows]
    hash_failures = []
    missing_files = []
    for row in rows:
        path = Path(row["source"]["path"])
        if not path.is_absolute():
            path = manifest.parent / path
        if not path.is_file():
            missing_files.append(row["item_id"])
        elif sha256(path) != row["source"]["sha256"]:
            hash_failures.append(row["item_id"])
    temperatures = [float(row["fact"]["true_value"]) for row in rows]
    distances = [float(row["fact"]["distance_to_scene_km"]) for row in rows]
    luminance = [float(row["source"]["mean_luminance"]) for row in rows]
    review_complete = sum(
        all(row.get("manual_review", {}).get(field) is not None for field in (
            "outdoor_scene", "location_credible", "carrier_region_approved"
        )) for row in rows
    )
    audit = {
        "schema_version": "cta/rvta-context-source-audit-v1",
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256(manifest),
        "items": len(rows),
        "unique_item_ids": len(set(item_ids)),
        "unique_fact_ids": len(set(fact_ids)),
        "unique_stations": len({row["fact"]["station_id"] for row in rows}),
        "station_counts": dict(sorted(Counter(row["fact"]["station_id"] for row in rows).items())),
        "license_counts": dict(sorted(Counter(row["source"]["license"] for row in rows).items())),
        "temperature_c": {
            "minimum": min(temperatures),
            "maximum": max(temperatures),
            "mean": statistics.fmean(temperatures),
            "median": statistics.median(temperatures),
        },
        "nearest_station_distance_km": {
            "minimum": min(distances),
            "maximum": max(distances),
            "mean": statistics.fmean(distances),
            "median": statistics.median(distances),
        },
        "mean_luminance": {
            "minimum": min(luminance),
            "maximum": max(luminance),
            "mean": statistics.fmean(luminance),
            "median": statistics.median(luminance),
        },
        "missing_files": missing_files,
        "hash_failures": hash_failures,
        "manual_review_complete_items": review_complete,
        "held_out_ready": (
            len(rows) == len(set(item_ids)) == len(set(fact_ids))
            and not missing_files and not hash_failures and review_complete == len(rows)
        ),
        "evidence_boundary": (
            "This audit covers source integrity and descriptive distributions only. It contains no victim "
            "inference and cannot support an attack-success claim. The candidate pool is not held-out ready "
            "until independent outdoor/location/carrier-region reviews are complete."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if missing_files or hash_failures or len(item_ids) != len(set(item_ids)) or len(fact_ids) != len(set(fact_ids)):
        raise RuntimeError("source integrity audit failed")


if __name__ == "__main__":
    main()
