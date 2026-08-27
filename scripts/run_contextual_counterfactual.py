#!/usr/bin/env python3
"""Run one LVLM on a frozen RVTA-Context v1 manifest."""

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

from cta.contextual_counterfactual import (
    CONDITIONS,
    exact_read_match,
    parse_semantic_answer,
    parse_temperature,
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
    if not manifest:
        raise ValueError("RVTA-Context manifest is empty")
    expected_items = int(config["expected_items"])
    item_ids = {row["item_id"] for row in manifest}
    expected_keys = {(row["item_id"], row["condition"]) for row in manifest}
    if len(item_ids) != expected_items:
        raise ValueError(f"expected {expected_items} items, found {len(item_ids)}")
    if {row["condition"] for row in manifest} != set(CONDITIONS):
        raise ValueError("condition mismatch")
    if len(expected_keys) != expected_items * len(CONDITIONS) or len(expected_keys) != len(manifest):
        raise ValueError("manifest coverage or uniqueness failed")
    for row in manifest:
        if file_sha256(Path(row["image_path"])) != row["image_sha256"]:
            raise ValueError(f"{row['item_id']}/{row['condition']}: rendered image hash mismatch")

    output_root = Path(config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "predictions.jsonl"
    existing = read_jsonl(output_path)
    completed = {(row["item_id"], row["condition"]) for row in existing}
    if not completed.issubset(expected_keys):
        raise ValueError("existing log contains keys outside the manifest")

    seed = int(config.get("seed", 20260827))
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
        "schema_version": "cta/rvta-context-run-v1",
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
        "seed": seed,
        "retry_policy": "none",
    }
    provenance_path = output_root / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    for source in tqdm(sorted(manifest, key=lambda row: (row["item_id"], row["condition"])), desc="RVTA-Context"):
        key = (source["item_id"], source["condition"])
        if key in completed:
            continue
        started = time.time()
        verify_raw = model.infer(source["image_path"], source["verification_question"])
        numeric_raw = model.infer(source["image_path"], source["numeric_question"])
        read_raw = model.infer(source["image_path"], source["read_prompt"])
        parsed = parse_semantic_answer(verify_raw, "yesno", "semantic")
        numeric_parsed = parse_temperature(numeric_raw)
        row = {
            **source,
            "verify_raw": verify_raw,
            "parsed_semantic": parsed,
            "answer_correct": parsed == source["correct_semantic"],
            "target_match": parsed == source["target_semantic"],
            "numeric_raw": numeric_raw,
            "numeric_parsed": numeric_parsed,
            "numeric_correct": (
                numeric_parsed is not None
                and abs(numeric_parsed - float(source["registered_true_value"])) <= 0.05
            ),
            "numeric_target_match": (
                numeric_parsed is not None
                and abs(numeric_parsed - float(source["registered_candidate_value"])) <= 0.05
            ),
            "read_raw": read_raw,
            "exact_read_match": exact_read_match(read_raw, source["registered_claim"]),
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
    provenance["status"] = "complete" if completed == expected_keys else "incomplete"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    if provenance["status"] != "complete":
        raise RuntimeError(f"run incomplete: {len(completed)}/{len(expected_keys)}")


if __name__ == "__main__":
    main()
