#!/usr/bin/env python3
"""Run one LVLM on a paired question-conditioned attack manifest."""

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
from cta.question_bench import (
    answer_score,
    file_sha256,
    normalize_answer,
    scenetap_compatible_score,
    summarize_question_rows,
    target_matches_any,
)
from cta.rio_bench import rio_mc_score


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


def safe_git_head() -> str:
    try:
        root = Path(__file__).resolve().parents[1]
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "not-a-git-checkout"


def save_summary(output_root: Path, rows: list[dict], threshold: float) -> None:
    summary = summarize_question_rows(rows, threshold)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["source_manifest"]).resolve()
    rows = read_jsonl(manifest_path)
    if not rows:
        raise ValueError("source manifest is empty")
    output_root = Path(config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    expected_questions = int(config.get("expected_questions", 0))
    question_ids = {row["question_id"] for row in rows}
    conditions = sorted({row["condition"] for row in rows})
    if expected_questions and len(question_ids) != expected_questions:
        raise ValueError(f"expected {expected_questions} question ids, found {len(question_ids)}")
    expected_keys = {(row["question_id"], row["condition"]) for row in rows}
    if len(expected_keys) != len(rows):
        raise ValueError("manifest contains duplicate question-condition keys")

    threshold = float(config.get("clean_correct_threshold", 1.0))
    scoring_profile = str(config.get("scoring_profile", "diagnostic_short_answer"))
    output_path = output_root / "predictions.jsonl"
    existing = read_jsonl(output_path)
    completed = {(row["question_id"], row["condition"]) for row in existing}
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
        "schema_version": "cta/question-benchmark-run-v1",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "git_head": safe_git_head(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "model": model.provenance(),
        "questions": len(question_ids),
        "conditions": conditions,
        "expected_rows": len(expected_keys),
        "query_policy": "one model call per question-condition; original question text; generation settings recorded in model provenance",
        "seed": seed,
        "clean_correct_threshold": threshold,
        "scoring_profile": scoring_profile,
        "metric_boundary": "SceneTAP-compatible TypoD/VQAv2 rules, RIO Obj-MC compatibility, or diagnostic normalized short-answer score; official benchmark claims require public scorer replay",
    }
    provenance_path = output_root / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    for source in tqdm(sorted(rows, key=lambda row: (row["question_id"], row["condition"])), desc="question eval"):
        key = (source["question_id"], source["condition"])
        if key in completed:
            continue
        started = time.time()
        raw = model.infer(source["image_path"], source["question"])
        prediction = raw.strip().splitlines()[0].strip()
        metadata = model.inference_metadata() if hasattr(model, "inference_metadata") else {}
        diagnostic_score = answer_score(prediction, source["answers"])
        if scoring_profile == "scenetap_public":
            primary_score = scenetap_compatible_score(
                prediction, source["answers"], source["question"], source["dataset"],
            )
        elif scoring_profile == "rio_obj_mc":
            primary_score = rio_mc_score(
                prediction, source["answers"][0], source["choices"],
            )
        elif scoring_profile == "diagnostic_short_answer":
            primary_score = diagnostic_score
        else:
            raise ValueError(f"unsupported scoring_profile: {scoring_profile}")
        row = {
            "question_id": source["question_id"], "dataset": source["dataset"],
            "condition": source["condition"], "question": source["question"],
            "answers": source["answers"], "target_answer": source["target_answer"],
            "target_content": source["target_content"], "target_aliases": source["target_aliases"],
            "task_type": source["task_type"], "source_sha256": source["source_sha256"],
            "image_path": source["image_path"], "image_sha256": source["image_sha256"],
            "overlay_text": source["overlay_text"], "raw_output": raw,
            "prediction": prediction, "normalized_prediction": normalize_answer(prediction),
            "answer_score": primary_score,
            "diagnostic_answer_score": diagnostic_score,
            "scoring_profile": scoring_profile,
            "target_match": target_matches_any(prediction, source["target_aliases"]),
            "inference_metadata": metadata,
            "latency_s": round(time.time() - started, 4),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        for optional_key in (
            "choices", "image_id", "attack_word", "rio_config", "rio_revision",
            "base_image_path", "base_image_sha256", "capture_profile",
            "capture_metadata", "official_attack_metadata",
        ):
            if optional_key in source:
                row[optional_key] = source[optional_key]
        append_jsonl(output_path, row)
        existing.append(row)
        completed.add(key)
        if len(existing) % 25 == 0:
            save_summary(output_root, existing, threshold)

    save_summary(output_root, existing, threshold)
    provenance["model"] = model.provenance()
    provenance["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["completed_rows"] = len(existing)
    provenance["status"] = "complete" if completed == expected_keys else "incomplete"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    if provenance["status"] != "complete":
        raise RuntimeError(f"run incomplete: {len(completed)}/{len(expected_keys)} rows")


if __name__ == "__main__":
    main()
