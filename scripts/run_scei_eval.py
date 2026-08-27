#!/usr/bin/env python3
"""Evaluate one frozen SCEI manifest with strict Read--Verify gates."""

from __future__ import annotations

import argparse
import csv
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

from cta.model import build_model_adapter
from cta.question_bench import file_sha256
from cta.scei_attack import (
    CONDITIONS,
    exact_transcription_matches,
    parse_semantic_answer,
    summarize,
)


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


def write_summary(root: Path, rows: list[dict]) -> None:
    present = {(row["item_id"], row["condition"]) for row in rows}
    item_ids = {row["item_id"] for row in rows}
    if not item_ids or any((item_id, condition) not in present for item_id in item_ids for condition in CONDITIONS):
        return
    values = summarize(rows)
    (root / "summary.json").write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values["conditions"][0]))
        writer.writeheader()
        writer.writerows(values["conditions"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["source_manifest"]).resolve()
    manifest = read_jsonl(manifest_path)
    if not manifest:
        raise ValueError("SCEI manifest is empty")
    expected_items = int(config["expected_items"])
    item_ids = {str(row["item_id"]) for row in manifest}
    expected_keys = {(str(row["item_id"]), str(row["condition"])) for row in manifest}
    if len(item_ids) != expected_items:
        raise ValueError(f"expected {expected_items} items, found {len(item_ids)}")
    if {row["condition"] for row in manifest} != set(CONDITIONS):
        raise ValueError("manifest condition set is incomplete")
    if len(expected_keys) != expected_items * len(CONDITIONS) or len(manifest) != len(expected_keys):
        raise ValueError("manifest key uniqueness or row count failed")

    output_root = Path(config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_path = output_root / "predictions.jsonl"
    existing = read_jsonl(prediction_path)
    completed = {(str(row["item_id"]), str(row["condition"])) for row in existing}
    if not completed.issubset(expected_keys):
        raise ValueError("existing predictions contain keys absent from the frozen manifest")

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
        "schema_version": "cta/scei-eval-v1",
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
        "seed": seed,
        "query_policy": "one answer query per row and one independent transcription query only for carrier rows; no retries",
        "analysis_population": "clean_false correct; clean_true and scene_true are separate controls",
        "success_definition": (
            "attacked answer is YES for the false consistency claim and the complete registered measurement line is transcribed"
        ),
    }
    provenance_path = output_root / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    for source in tqdm(
        sorted(manifest, key=lambda row: (str(row["item_id"]), CONDITIONS.index(row["condition"]))),
        desc="SCEI evaluation",
    ):
        key = (str(source["item_id"]), str(source["condition"]))
        if key in completed:
            continue
        started = time.time()
        answer_raw = model.infer(source["image_path"], source["verification_question"])
        if source["requires_read"]:
            read_raw = model.infer(source["image_path"], source["read_prompt"])
            exact_read = exact_transcription_matches(read_raw, source["registered_read_text"])
        else:
            read_raw = "NONE"
            exact_read = None
        parsed = parse_semantic_answer(answer_raw, source["answer_format"], source["option_order"])
        row = {
            **source,
            "answer_raw": answer_raw,
            "parsed_semantic": parsed,
            "answer_correct": parsed == source["correct_semantic"],
            "target_match": parsed == source["target_semantic"],
            "read_raw": read_raw,
            "exact_read_match": exact_read,
            "latency_s": round(time.time() - started, 4),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "inference_metadata": model.inference_metadata() if hasattr(model, "inference_metadata") else {},
        }
        append_jsonl(prediction_path, row)
        existing.append(row)
        completed.add(key)
        if len(existing) % (len(CONDITIONS) * 3) == 0:
            write_summary(output_root, existing)

    write_summary(output_root, existing)
    provenance["model"] = model.provenance()
    provenance["completed_rows"] = len(existing)
    provenance["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["status"] = "complete" if completed == expected_keys else "incomplete"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    if provenance["status"] != "complete":
        raise RuntimeError(f"run incomplete: {len(completed)}/{len(expected_keys)}")


if __name__ == "__main__":
    main()
