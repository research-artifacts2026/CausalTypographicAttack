#!/usr/bin/env python3
"""Freeze disjoint remaining-row shards after an interrupted ContraLedger run."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import file_sha256


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prefix-log", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    args = parser.parse_args()
    if args.num_shards < 2:
        raise ValueError("num-shards must be at least two")

    manifest_path = args.manifest.resolve()
    prefix_path = args.prefix_log.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite shard root: {output}")
    manifest_rows = read_jsonl(manifest_path)
    expected = {(str(row["item_id"]), str(row["condition"])): row for row in manifest_rows}
    if len(expected) != len(manifest_rows):
        raise ValueError("full manifest contains duplicate keys")
    prefix_rows = read_jsonl(prefix_path)
    prefix = {(str(row["item_id"]), str(row["condition"])): row for row in prefix_rows}
    if len(prefix) != len(prefix_rows) or not set(prefix).issubset(expected):
        raise ValueError("prefix contains duplicate or unregistered keys")
    for key, row in prefix.items():
        frozen = expected[key]
        if row.get("image_sha256") != frozen.get("image_sha256") or row.get("question") != frozen.get("question"):
            raise ValueError(f"prefix changed a frozen field: {key}")

    missing = [expected[key] for key in sorted(set(expected) - set(prefix))]
    if not missing:
        raise ValueError("prefix is already complete")
    shards = [[] for _ in range(args.num_shards)]
    for index, row in enumerate(missing):
        shards[index % args.num_shards].append(row)
    output.mkdir(parents=True)
    prefix_copy = output / "prefix_predictions.jsonl"
    shutil.copy2(prefix_path, prefix_copy)
    shard_records = []
    for index, rows in enumerate(shards):
        manifest = output / f"shard_{index}" / "manifest.jsonl"
        write_jsonl(manifest, rows)
        shard_records.append({
            "shard_index": index,
            "rows": len(rows),
            "manifest": str(manifest),
            "manifest_sha256": file_sha256(manifest),
            "output_root": str(output / f"shard_{index}" / "run"),
        })
    provenance = {
        "schema_version": "cta/contraledger-remaining-shards-v1",
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "full_manifest": str(manifest_path),
        "full_manifest_sha256": file_sha256(manifest_path),
        "expected_rows": len(expected),
        "prefix_source": str(prefix_path),
        "prefix_copy": str(prefix_copy),
        "prefix_rows": len(prefix),
        "prefix_sha256": file_sha256(prefix_copy),
        "missing_rows": len(missing),
        "partition": "round-robin over sorted missing (item_id, condition) keys",
        "shards": shard_records,
    }
    (output / "shard_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
