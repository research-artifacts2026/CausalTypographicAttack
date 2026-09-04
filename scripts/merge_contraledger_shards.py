#!/usr/bin/env python3
"""Merge a frozen ContraLedger prefix and complete disjoint row shards."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.contraledger import CONDITIONS, summarize
from cta.question_bench import file_sha256
from scripts.analyze_contraledger import _FROZEN_INPUT_FIELDS


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def git_head() -> str:
    try:
        root = Path(__file__).resolve().parents[1]
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "not-a-git-checkout"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    shard_root = args.shard_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite merged output: {output}")
    manifest_rows = read_jsonl(manifest_path)
    manifest = {(str(row["item_id"]), str(row["condition"])): row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("full manifest contains duplicate keys")
    shard_provenance_path = shard_root / "shard_provenance.json"
    shard_provenance = json.loads(shard_provenance_path.read_text(encoding="utf-8"))
    if shard_provenance.get("status") != "frozen":
        raise ValueError("shard plan is not frozen")
    if shard_provenance.get("full_manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("shard plan references a different full manifest")

    component_records = []
    prefix_path = Path(shard_provenance["prefix_copy"])
    if file_sha256(prefix_path) != shard_provenance["prefix_sha256"]:
        raise ValueError("frozen prefix changed")
    rows = read_jsonl(prefix_path)
    for shard in shard_provenance["shards"]:
        shard_manifest = Path(shard["manifest"])
        if file_sha256(shard_manifest) != shard["manifest_sha256"]:
            raise ValueError(f"shard {shard['shard_index']} manifest changed")
        run_root = Path(shard["output_root"])
        provenance_path = run_root / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        prediction_path = run_root / "predictions.jsonl"
        shard_rows = read_jsonl(prediction_path)
        if provenance.get("status") != "complete" or provenance.get("completed_rows") != shard["rows"]:
            raise ValueError(f"shard {shard['shard_index']} is incomplete")
        if provenance.get("manifest_sha256") != shard["manifest_sha256"]:
            raise ValueError(f"shard {shard['shard_index']} run references another manifest")
        rows.extend(shard_rows)
        component_records.append({
            "shard_index": shard["shard_index"],
            "prediction_log": str(prediction_path),
            "prediction_log_sha256": file_sha256(prediction_path),
            "provenance": str(provenance_path),
            "provenance_sha256": file_sha256(provenance_path),
        })

    keyed = {}
    for row in rows:
        key = (str(row["item_id"]), str(row["condition"]))
        if key in keyed:
            raise ValueError(f"duplicate merged key: {key}")
        if key not in manifest:
            raise ValueError(f"unregistered merged key: {key}")
        frozen = manifest[key]
        for field in _FROZEN_INPUT_FIELDS:
            if row.get(field) != frozen.get(field):
                raise ValueError(f"{key}: frozen field changed: {field}")
        keyed[key] = row
    if set(keyed) != set(manifest):
        raise ValueError(f"merged log is missing {len(set(manifest) - set(keyed))} keys")
    order = {condition: index for index, condition in enumerate(CONDITIONS)}
    merged = [keyed[key] for key in sorted(keyed, key=lambda key: (key[0], order[key[1]]))]
    summary = summarize(merged)
    output.mkdir(parents=True)
    prediction_path = output / "predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in merged),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": "cta/contraledger-merged-run-v1",
        "status": "complete",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "git_head": git_head(),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "shard_provenance": str(shard_provenance_path),
        "shard_provenance_sha256": file_sha256(shard_provenance_path),
        "components": component_records,
        "completed_rows": len(merged),
        "prediction_log_sha256": file_sha256(prediction_path),
        "merge_policy": "exact union by registered key; duplicate, missing, extra, and frozen-field mismatches are fatal",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(merged), "sha256": provenance["prediction_log_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
