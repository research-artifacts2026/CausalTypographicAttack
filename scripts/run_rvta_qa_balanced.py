#!/usr/bin/env python3
"""Run one LVLM on a frozen balanced RVTA-QA manifest."""

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
from cta.rvta_qa_balanced import parse_semantic_answer, summarize, transcription_matches


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
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={Path(__file__).resolve().parents[1]}", "-C",
             str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "not-a-git-checkout"


def write_summary(root: Path, rows: list[dict]) -> None:
    values = summarize(rows)
    (root / "summary.json").write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    pooled = values["pooled"]
    if pooled:
        with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pooled[0]))
            writer.writeheader()
            writer.writerows(pooled)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["source_manifest"]).resolve()
    manifest = read_jsonl(manifest_path)
    if not manifest:
        raise ValueError("balanced RVTA-QA manifest is empty")
    expected_items = int(config["expected_items"])
    item_ids = {row["item_id"] for row in manifest}
    expected_keys = {(row["item_id"], row["condition"]) for row in manifest}
    if len(item_ids) != expected_items or len(expected_keys) != len(manifest):
        raise ValueError("manifest item count or key uniqueness check failed")
    output_root = Path(config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "predictions.jsonl"
    existing = read_jsonl(output_path)
    completed = {(row["item_id"], row["condition"]) for row in existing}
    seed = int(config.get("seed", 42))
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
        "schema_version": "cta/rvta-qa-balanced-run-v1",
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
        "queries_per_row": 2,
        "metric_boundary": "semantic clean-conditioned target ASR; grounded endpoint requires registered claim transcription",
        "seed": seed,
    }
    provenance_path = output_root / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    for source in tqdm(sorted(manifest, key=lambda row: (row["item_id"], row["condition"])), desc="RVTA-QA balanced"):
        key = (source["item_id"], source["condition"])
        if key in completed:
            continue
        started = time.time()
        answer_raw = model.infer(source["image_path"], source["verification_question"])
        read_raw = model.infer(source["image_path"], source["read_prompt"])
        parsed = parse_semantic_answer(answer_raw, source["answer_format"], source["option_order"])
        copy_keys = (
            "item_id", "dataset", "condition", "target_label", "family", "source_path",
            "source_sha256", "image_path", "image_sha256", "proposition_truth",
            "verification_claim", "attack_claim", "benign_control_claim",
            "ordinary_world_assumption", "verification_question", "answer_format",
            "option_order", "counterbalance_cell", "correct_semantic", "target_semantic",
            "correct_answer_token", "target_answer_token", "overlay_text",
            "registered_read_text", "bbox", "placement", "overlay_area_fraction", "stage",
        )
        row = {
            **{name: source[name] for name in copy_keys},
            "upstream_source_sha256": source.get("upstream_source_sha256", source["source_sha256"]),
            "source_reencoded": bool(source.get("source_reencoded", False)),
            "answer_raw": answer_raw,
            "parsed_semantic": parsed,
            "answer_correct": parsed == source["correct_semantic"],
            "target_match": parsed == source["target_semantic"],
            "read_raw": read_raw,
            "read_match": transcription_matches(read_raw, source["registered_read_text"]),
            "latency_s": round(time.time() - started, 4),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "inference_metadata": model.inference_metadata() if hasattr(model, "inference_metadata") else {},
        }
        append_jsonl(output_path, row)
        existing.append(row)
        completed.add(key)
        if len(existing) % 25 == 0:
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

