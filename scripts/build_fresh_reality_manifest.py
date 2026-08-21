#!/usr/bin/env python3
"""Materialize a fresh, high-visibility COCO pilot outside registered prior splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.reality_violation import violation_family


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excluded_ids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            excluded.update(str(row["sample_id"]) for row in value)
            continue
        for key, items in value.items():
            if key.endswith("_ids") and isinstance(items, list):
                excluded.update(str(item) for item in items)
    return excluded


def metadata_candidates(shards: list[Path], excluded: set[str], minimum_area: float) -> list[dict]:
    candidates: list[dict] = []
    columns = ["image_id", "file_name", "width", "height", "annotations"]
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=64, columns=columns):
            for row in batch.to_pylist():
                sample_id = f"coco-{int(row['image_id']):012d}"
                if sample_id in excluded:
                    continue
                ann = row["annotations"]
                valid = [index for index, crowd in enumerate(ann["iscrowd"]) if int(crowd) == 0]
                if not valid:
                    continue
                best = max(valid, key=lambda index: float(ann["area"][index]))
                area = float(ann["area"][best]) / (float(row["width"]) * float(row["height"]))
                if area < minimum_area:
                    continue
                label = str(ann["category_name"][best])
                candidates.append({
                    "image_id": int(row["image_id"]),
                    "sample_id": sample_id,
                    "file_name": str(row["file_name"]),
                    "target_label": label,
                    "target_class_id": int(ann["category_id"][best]),
                    "target_area": area,
                    "target_bbox": [float(value) for value in ann["bbox"][best]],
                    "target_bbox_area": (
                        float(ann["bbox"][best][2]) * float(ann["bbox"][best][3])
                        / (float(row["width"]) * float(row["height"]))
                    ),
                    "labels": sorted({str(ann["category_name"][index]) for index in valid}),
                    "violation_family": violation_family(label),
                })
    return candidates


def select_diverse(candidates: list[dict], n: int, seed: int) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in candidates:
        grouped.setdefault(row["violation_family"], []).append(row)
    for family, rows in grouped.items():
        grouped[family] = sorted(
            rows,
            key=lambda row: hashlib.sha256(f"{seed}:{family}:{row['sample_id']}".encode()).hexdigest(),
        )
    selected: list[dict] = []
    cursor = 0
    while len(selected) < n:
        progressed = False
        for family in sorted(grouped):
            if cursor < len(grouped[family]):
                selected.append(grouped[family][cursor])
                progressed = True
                if len(selected) == n:
                    break
        if not progressed:
            break
        cursor += 1
    if len(selected) < n:
        raise ValueError(f"requested {n} samples but only selected {len(selected)}")
    return selected


def materialize(shards: list[Path], selected: list[dict], image_dir: Path) -> list[dict]:
    by_id = {int(row["image_id"]): row for row in selected}
    image_dir.mkdir(parents=True, exist_ok=True)
    found: dict[int, dict] = {}
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=32, columns=["image", "image_id"]):
            for row in batch.to_pylist():
                image_id = int(row["image_id"])
                if image_id not in by_id:
                    continue
                image_bytes = row["image"]["bytes"]
                output = image_dir / f"{image_id:012d}.jpg"
                if not output.exists() or sha256(output) != hashlib.sha256(image_bytes).hexdigest():
                    output.write_bytes(image_bytes)
                record = dict(by_id[image_id])
                record.pop("image_id")
                record["image_path"] = str(output.resolve())
                record["source_sha256"] = sha256(output)
                found[image_id] = record
    missing = set(by_id) - set(found)
    if missing:
        raise RuntimeError(f"selected image bytes were not found: {sorted(missing)}")
    return [found[int(row["image_id"])] for row in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--minimum-target-area", type=float, default=0.45)
    args = parser.parse_args()

    shards = sorted((args.dataset_root / "data").glob("validation-*.parquet"))
    if not shards:
        raise FileNotFoundError("no COCO validation parquet shards found")
    excluded = excluded_ids(args.exclude_manifest)
    candidates = metadata_candidates(shards, excluded, args.minimum_target_area)
    selected = select_diverse(candidates, args.n, args.seed)
    rows = materialize(shards, selected, args.image_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": "cta/fresh-reality-pilot-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_seed": args.seed,
        "minimum_target_area": args.minimum_target_area,
        "exclusion_manifests": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in args.exclude_manifest
        ],
        "excluded_prior_ids": len(excluded),
        "candidate_count": len(candidates),
        "selected_ids": [row["sample_id"] for row in rows],
        "selected_families": [row["violation_family"] for row in rows],
        "output_manifest": str(args.output.resolve()),
        "output_manifest_sha256": sha256(args.output),
        "parquet_shards": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in shards],
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": len(rows), "ids": provenance["selected_ids"], "families": provenance["selected_families"]}, indent=2))


if __name__ == "__main__":
    main()
