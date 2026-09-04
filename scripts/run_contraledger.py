#!/usr/bin/env python3
"""Run one LVLM on a frozen ContraLedger manifest."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.contraledger import CONDITIONS, exact_transcription_matches, parse_answer, summarize
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
    if not manifest:
        raise ValueError("ContraLedger manifest is empty")
    expected_items = int(config["expected_items"])
    item_ids = {str(row["item_id"]) for row in manifest}
    expected_keys = {(str(row["item_id"]), str(row["condition"])) for row in manifest}
    if len(item_ids) != expected_items or len(expected_keys) != expected_items * len(CONDITIONS):
        raise ValueError("manifest item/condition coverage mismatch")
    if {row["condition"] for row in manifest} != set(CONDITIONS):
        raise ValueError("manifest condition names mismatch")

    output_root = Path(config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "predictions.jsonl"
    existing = read_jsonl(output_path)
    completed = {(str(row["item_id"]), str(row["condition"])) for row in existing}
    if not completed.issubset(expected_keys):
        raise ValueError("existing log contains unregistered keys")

    seed = int(config.get("seed", 20260904))
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    model = build_model_adapter(config["model"])
    provenance = {
        "schema_version": "cta/contraledger-run-v1",
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
        "items": len(item_ids),
        "rows": len(manifest),
        "queries_per_row": 3,
        "probe_policy": "independent Read, Knowledge, and Decide calls; no conversational carry-over",
        "retry_policy": "none",
        "seed": seed,
    }
    provenance_path = output_root / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    ordered = sorted(manifest, key=lambda row: (str(row["item_id"]), CONDITIONS.index(row["condition"])))
    for source in tqdm(ordered, desc="ContraLedger-v1"):
        key = (str(source["item_id"]), str(source["condition"]))
        if key in completed:
            continue
        started = time.time()
        read_raw = model.infer(source["image_path"], source["probe_prompts"]["read"])
        knowledge_raw = model.infer(source["source_path"], source["probe_prompts"]["knowledge"])
        decide_raw = model.infer(source["image_path"], source["probe_prompts"]["decide"])
        row = {
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
