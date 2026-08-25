#!/usr/bin/env python3
"""Merge a frozen balanced prefix and completed disjoint shards."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import file_sha256
from cta.rvta_qa_balanced import summarize


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
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
        raise FileExistsError(f"refusing to overwrite merge output: {output}")
    manifest = read_jsonl(manifest_path)
    expected = {(row["item_id"], row["condition"]): row for row in manifest}
    shard_provenance_path = shard_root / "shard_provenance.json"
    shard_provenance = json.loads(shard_provenance_path.read_text(encoding="utf-8"))
    prefix_path = Path(shard_provenance["prefix_copy"])
    component_paths = [prefix_path]
    component_provenance = []
    rows = read_jsonl(prefix_path)
    for shard in shard_provenance["shards"]:
        run_root = Path(shard["output_root"])
        run_provenance_path = run_root / "provenance.json"
        run_provenance = json.loads(run_provenance_path.read_text(encoding="utf-8"))
        prediction_path = run_root / "predictions.jsonl"
        shard_rows = read_jsonl(prediction_path)
        if run_provenance.get("status") != "complete" or run_provenance.get("completed_rows") != shard["rows"]:
            raise ValueError(f"shard {shard['shard_index']} is incomplete")
        if file_sha256(Path(shard["manifest"])) != shard["manifest_sha256"]:
            raise ValueError(f"shard {shard['shard_index']} manifest changed")
        rows.extend(shard_rows)
        component_paths.append(prediction_path)
        component_provenance.append({
            "shard_index": shard["shard_index"],
            "prediction_log": str(prediction_path),
            "prediction_log_sha256": file_sha256(prediction_path),
            "provenance": str(run_provenance_path),
            "provenance_sha256": file_sha256(run_provenance_path),
        })
    keyed = {}
    for row in rows:
        key = (row["item_id"], row["condition"])
        if key in keyed:
            raise ValueError(f"duplicate merged key: {key}")
        if key not in expected or row.get("image_sha256") != expected[key].get("image_sha256"):
            raise ValueError(f"unexpected key or image hash mismatch: {key}")
        keyed[key] = row
    if set(keyed) != set(expected):
        missing = set(expected) - set(keyed)
        raise ValueError(f"merged log missing {len(missing)} keys")
    merged = [keyed[key] for key in sorted(keyed)]
    output.mkdir(parents=True)
    prediction_path = output / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    values = summarize(merged)
    (output / "summary.json").write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values["pooled"][0]))
        writer.writeheader()
        writer.writerows(values["pooled"])
    provenance = {
        "schema_version": "cta/rvta-qa-balanced-merged-run-v1",
        "status": "complete",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "git_head": git_head(),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "shard_provenance": str(shard_provenance_path),
        "shard_provenance_sha256": file_sha256(shard_provenance_path),
        "components": component_provenance,
        "completed_rows": len(merged),
        "prediction_log_sha256": file_sha256(prediction_path),
        "merge_policy": "exact union by (item_id, condition); duplicate, missing, extra, and image-hash mismatches are fatal",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(merged), "sha256": provenance["prediction_log_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
