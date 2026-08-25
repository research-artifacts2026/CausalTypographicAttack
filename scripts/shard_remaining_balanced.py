#!/usr/bin/env python3
"""Freeze disjoint item-level shards for the unfinished rows of a balanced run."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import file_sha256


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--existing-log", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    if args.shards < 2:
        raise ValueError("--shards must be at least 2")

    manifest_path = args.manifest.resolve()
    existing_path = args.existing_log.resolve()
    config_path = args.base_config.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite shard root: {output}")
    manifest = read_jsonl(manifest_path)
    existing = read_jsonl(existing_path)
    expected = {(row["item_id"], row["condition"]) for row in manifest}
    completed = {(row["item_id"], row["condition"]) for row in existing}
    if len(expected) != len(manifest) or len(completed) != len(existing):
        raise ValueError("manifest or existing log contains duplicate keys")
    if not completed.issubset(expected):
        raise ValueError("existing log contains keys outside the frozen manifest")
    by_key = {(row["item_id"], row["condition"]): row for row in manifest}
    for row in existing:
        key = (row["item_id"], row["condition"])
        if row.get("image_sha256") != by_key[key].get("image_sha256"):
            raise ValueError(f"image hash mismatch for completed key {key}")

    output.mkdir(parents=True)
    prefix_copy = output / "prefix_predictions.jsonl"
    shutil.copyfile(existing_path, prefix_copy)
    remaining = [row for row in manifest if (row["item_id"], row["condition"]) not in completed]
    item_ids = sorted({row["item_id"] for row in remaining})
    assignments = {item_id: index % args.shards for index, item_id in enumerate(item_ids)}
    base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    shard_records = []
    for shard_index in range(args.shards):
        rows = [row for row in remaining if assignments[row["item_id"]] == shard_index]
        rows.sort(key=lambda row: (row["item_id"], row["condition"]))
        shard_manifest = output / f"manifest_shard_{shard_index}.jsonl"
        write_jsonl(shard_manifest, rows)
        shard_output = output / f"run_shard_{shard_index}"
        shard_config = dict(base_config)
        shard_config["source_manifest"] = str(shard_manifest)
        shard_config["output_root"] = str(shard_output)
        shard_config["expected_items"] = len({row["item_id"] for row in rows})
        shard_config_path = output / f"config_shard_{shard_index}.yaml"
        shard_config_path.write_text(yaml.safe_dump(shard_config, sort_keys=False), encoding="utf-8")
        shard_records.append({
            "shard_index": shard_index,
            "items": shard_config["expected_items"],
            "rows": len(rows),
            "manifest": str(shard_manifest),
            "manifest_sha256": file_sha256(shard_manifest),
            "config": str(shard_config_path),
            "config_sha256": file_sha256(shard_config_path),
            "output_root": str(shard_output),
        })
    flattened = {
        (row["item_id"], row["condition"])
        for record in shard_records for row in read_jsonl(Path(record["manifest"]))
    }
    if flattened != expected - completed:
        raise RuntimeError("shards do not exactly partition unfinished keys")
    provenance = {
        "schema_version": "cta/rvta-qa-balanced-shards-v1",
        "status": "frozen",
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "existing_log": str(existing_path),
        "existing_log_sha256": file_sha256(existing_path),
        "prefix_copy": str(prefix_copy),
        "prefix_copy_sha256": file_sha256(prefix_copy),
        "completed_rows": len(completed),
        "remaining_rows": len(remaining),
        "shards": shard_records,
    }
    (output / "shard_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
