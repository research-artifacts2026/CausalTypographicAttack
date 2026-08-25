#!/usr/bin/env python3
"""Freeze print/display assets and a preregistered physical-capture schedule."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_METHODS = (
    "no_attack",
    "naive_typography",
    "rio_typography_hard",
    "evidence_cta",
    "rio_scenetap_hard",
)

VIEWS = (
    {"view": "frontal_075m", "distance_m": "0.75", "yaw_deg": "0", "lighting": "standard"},
    {"view": "oblique_150m", "distance_m": "1.50", "yaw_deg": "25", "lighting": "standard"},
    {"view": "frontal_lowlight_150m", "distance_m": "1.50", "yaw_deg": "0", "lighting": "low"},
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--questions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    args = parser.parse_args()
    if args.questions <= 0:
        raise ValueError("--questions must be positive")

    manifest = args.manifest.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty kit: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(manifest)
    by_key = {(str(row["question_id"]), str(row["condition"])): row for row in rows}
    clean_order = []
    seen = set()
    for row in rows:
        qid = str(row["question_id"])
        if row["condition"] == "no_attack" and qid not in seen:
            clean_order.append(qid)
            seen.add(qid)
    selected = clean_order[: args.questions]
    if len(selected) != args.questions:
        raise ValueError(f"manifest exposes only {len(selected)} clean questions")
    missing = [(qid, method) for qid in selected for method in args.methods if (qid, method) not in by_key]
    if missing:
        raise ValueError(f"capture methods are incomplete; first missing keys: {missing[:5]}")

    assets = []
    for qid in selected:
        for method in args.methods:
            source = Path(by_key[(qid, method)]["image_path"]).resolve()
            suffix = source.suffix.lower() or ".jpg"
            destination = output_root / "assets" / method / f"{qid}{suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            source_hash = sha256(source)
            copied_hash = sha256(destination)
            if source_hash != copied_hash:
                raise AssertionError(f"asset copy hash mismatch: {source}")
            assets.append({
                "question_id": qid,
                "method": method,
                "source_image_path": str(source),
                "source_attack_sha256": source_hash,
                "asset_path": str(destination),
                "asset_sha256": copied_hash,
                "question": by_key[(qid, method)]["question"],
                "answer": by_key[(qid, method)].get(
                    "answer", by_key[(qid, method)].get("answers", [])
                ),
            })

    rng = random.Random(args.seed)
    schedule = []
    for qid in selected:
        methods = list(args.methods)
        rng.shuffle(methods)
        for order, method in enumerate(methods, start=1):
            asset = next(row for row in assets if row["question_id"] == qid and row["method"] == method)
            for view in VIEWS:
                capture_id = f"tier1__{qid}__{method}__{view['view']}"
                schedule.append({
                    "capture_id": capture_id,
                    "scene_id": f"display-{qid}",
                    "question_id": qid,
                    "method": method,
                    "tier": "tier1_physical_recapture",
                    "method_order": order,
                    **view,
                    "camera_model": "",
                    "capture_time": "",
                    "operator_id": "",
                    "source_attack_sha256": asset["source_attack_sha256"],
                    "asset_path": asset["asset_path"],
                    "photo_path": "",
                    "photo_sha256": "",
                    "status": "pending_capture",
                })

    asset_fields = list(assets[0])
    schedule_fields = list(schedule[0])
    write_csv(output_root / "asset_manifest.csv", assets, asset_fields)
    write_csv(output_root / "capture_manifest.csv", schedule, schedule_fields)
    provenance = {
        "schema_version": "cta/physical-capture-kit-v1",
        "status": "prepared_not_captured",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256(manifest),
        "questions": len(selected),
        "methods": list(args.methods),
        "views": list(VIEWS),
        "assets": len(assets),
        "scheduled_photographs": len(schedule),
        "seed": args.seed,
        "evidence_boundary": "This artifact is a preregistered capture kit. It contains no physical photographs and cannot support a physical-world result until the manifest is completed and validated.",
    }
    (output_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (output_root / "README.md").write_text(
        "# Frozen physical-capture kit\n\n"
        "This directory contains immutable display/print assets and a randomized capture schedule. "
        "It is **not** a physical result. Fill every blank camera/operator/photo field in "
        "`capture_manifest.csv`, retain camera originals, and then run "
        "`scripts/validate_physical_capture.py`. Tier 1 may only be described as physical recapture.\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
