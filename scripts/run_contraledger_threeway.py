#!/usr/bin/env python3
"""Run one LVLM on the frozen three-state ContraLedger confirmation set."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.contraledger_threeway import (
    CONDITIONS,
    exact_read,
    parse_choice,
    parse_knowledge,
    summarize,
)
from cta.model import build_model_adapter
from cta.question_bench import file_sha256


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_head() -> str:
    try:
        root = Path(__file__).resolve().parents[1]
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "not-a-git-checkout"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["source_manifest"]).resolve()
    manifest = read_jsonl(manifest_path)
    expected_items = int(config["expected_items"])
    expected_keys = {(str(row["item_id"]), str(row["condition"])) for row in manifest}
    if len(expected_keys) != expected_items * len(CONDITIONS):
        raise ValueError("three-way manifest coverage mismatch")
    if {row["condition"] for row in manifest} != set(CONDITIONS):
        raise ValueError("three-way condition names mismatch")

    output_root = Path(config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "predictions.jsonl"
    existing = read_jsonl(output_path)
    completed = {(str(row["item_id"]), str(row["condition"])) for row in existing}
    if not completed.issubset(expected_keys) or len(completed) != len(existing):
        raise ValueError("existing log contains duplicate or unregistered keys")

    model = build_model_adapter(config["model"])
    provenance = {
        "schema_version": "cta/contraledger-threeway-run-v1",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "git_head": git_head(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "model": model.provenance(),
        "items": expected_items,
        "rows": len(manifest),
        "probe_policy": "Decide on all rows; independent Read and Knowledge on false-record rows only",
        "retry_policy": "none",
    }
    provenance_path = output_root / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    order = {name: index for index, name in enumerate(CONDITIONS)}
    for source in tqdm(sorted(manifest, key=lambda row: (str(row["item_id"]), order[row["condition"]])), desc="ContraLedger-3way"):
        key = (str(source["item_id"]), str(source["condition"]))
        if key in completed:
            continue
        decide_raw = model.infer(source["image_path"], source["probe_prompts"]["decide"])
        row = {
            **source,
            "decide_raw": decide_raw,
            "decide_parsed": parse_choice(decide_raw, source["option_map"]),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "inference_metadata": model.inference_metadata() if hasattr(model, "inference_metadata") else {},
        }
        if source["condition"] == "record_false":
            read_raw = model.infer(source["image_path"], source["probe_prompts"]["read"])
            knowledge_raw = model.infer(source["source_path"], source["probe_prompts"]["knowledge"])
            row.update({
                "read_raw": read_raw,
                "read_match": exact_read(read_raw, source["registered_read_text"]),
                "knowledge_raw": knowledge_raw,
                "knowledge_parsed": parse_knowledge(knowledge_raw, source["knowledge_option_order"]),
            })
        append_jsonl(output_path, row)
        existing.append(row)
        completed.add(key)

    summary = summarize(existing)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    provenance["completed_rows"] = len(existing)
    provenance["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["model"] = model.provenance()
    provenance["status"] = "complete" if completed == expected_keys else "incomplete"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    if provenance["status"] != "complete":
        raise RuntimeError(f"run incomplete: {len(completed)}/{len(expected_keys)}")


if __name__ == "__main__":
    main()
