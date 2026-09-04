#!/usr/bin/env python3
"""Run an arbitrary frozen subset of registered ContraLedger rows."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.contraledger import exact_transcription_matches, parse_answer
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
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "not-a-git-checkout"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    config_path = args.model_config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows = read_jsonl(manifest_path)
    expected = {(str(row["item_id"]), str(row["condition"])) for row in rows}
    if not rows or len(expected) != len(rows):
        raise ValueError("shard manifest is empty or contains duplicate keys")

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions.jsonl"
    existing = read_jsonl(prediction_path)
    completed = {(str(row["item_id"]), str(row["condition"])) for row in existing}
    if len(completed) != len(existing) or not completed.issubset(expected):
        raise ValueError("existing shard log has duplicate or unregistered keys")

    model = build_model_adapter(config["model"])
    provenance = {
        "schema_version": "cta/contraledger-row-shard-run-v1",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_config": str(config_path),
        "model_config_sha256": file_sha256(config_path),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "git_head": git_head(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "model": model.provenance(),
        "rows": len(rows),
        "probe_policy": "independent Read, Knowledge, and Decide calls",
        "retry_policy": "none",
    }
    provenance_path = output / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    for source in tqdm(sorted(rows, key=lambda row: (str(row["item_id"]), str(row["condition"]))), desc="ContraLedger-shard"):
        key = (str(source["item_id"]), str(source["condition"]))
        if key in completed:
            continue
        started = time.time()
        read_raw = model.infer(source["image_path"], source["probe_prompts"]["read"])
        knowledge_raw = model.infer(source["source_path"], source["probe_prompts"]["knowledge"])
        decide_raw = model.infer(source["image_path"], source["probe_prompts"]["decide"])
        result = {
            **source,
            "read_raw": read_raw,
            "read_match": exact_transcription_matches(read_raw, source["registered_read_text"]),
            "knowledge_raw": knowledge_raw,
            "knowledge_parsed": parse_answer(knowledge_raw, source["option_order"]),
            "decide_raw": decide_raw,
            "decide_parsed": parse_answer(decide_raw, source["option_order"]),
            "latency_s": round(time.time() - started, 4),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "inference_metadata": model.inference_metadata() if hasattr(model, "inference_metadata") else {},
        }
        append_jsonl(prediction_path, result)
        completed.add(key)
    provenance["completed_rows"] = len(completed)
    provenance["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["model"] = model.provenance()
    provenance["status"] = "complete" if completed == expected else "incomplete"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    if provenance["status"] != "complete":
        raise RuntimeError(f"shard incomplete: {len(completed)}/{len(expected)}")


if __name__ == "__main__":
    main()
