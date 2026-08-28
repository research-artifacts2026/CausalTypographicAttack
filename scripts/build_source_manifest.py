#!/usr/bin/env python3
"""Freeze a model-free source-image manifest for downstream dataset builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.data import load_dataset
from cta.question_bench import file_sha256


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"]).resolve()
    manifest_path = output_root / "sample_manifest.json"
    provenance_path = output_root / "provenance.json"
    if manifest_path.exists() or provenance_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen source manifest: {output_root}")
    samples = load_dataset(
        str(config["dataset_name"]),
        config["dataset_root"],
        int(config["num_samples"]),
        int(config["seed"]),
        dataset_split=config.get("dataset_split"),
    )
    rows = [sample.to_dict() for sample in samples]
    _write_json(manifest_path, rows)
    provenance = {
        "schema_version": "cta/source-manifest-v1",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "dataset_name": str(config["dataset_name"]),
        "dataset_root": str(Path(config["dataset_root"]).resolve()),
        "dataset_split": config.get("dataset_split"),
        "seed": int(config["seed"]),
        "items": len(rows),
        "unique_item_ids": len({row["sample_id"] for row in rows}),
        "target_label_counts": dict(sorted(Counter(row["target_label"] for row in rows).items())),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "selection_rule": "dataset loader shuffle by fixed seed; no planner or victim outputs consulted",
        "source_hash_aggregate": hashlib.sha256(
            "\n".join(row["source_sha256"] for row in rows).encode("utf-8")
        ).hexdigest(),
    }
    _write_json(provenance_path, provenance)
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
