#!/usr/bin/env python3
"""Run one frozen LVLM on Experiment-A mechanism controls.

The runner requires an explicit preregistered manifest hash and refuses sampled
decoding, stale images, changed renderer code, incomplete condition sets, or a
resume log inconsistent with the frozen manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.bridge_mechanism_controls import (
    ALL_CONDITIONS,
    SCHEMA_VERSION,
    parse_semantic_answer,
    read_jsonl,
    summarize_conditions,
    transcription_fields_match,
    validate_manifest_rows,
)
from cta.question_bench import file_sha256


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


def model_snapshot_descriptor(cfg: dict) -> dict:
    root = Path(str(cfg["name_or_path"])).resolve()
    metadata_hashes = {}
    inventory = []
    if root.is_dir():
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
            relative = str(path.relative_to(root)).replace("\\", "/")
            size = path.stat().st_size
            inventory.append(f"{relative}\t{size}")
            if path.suffix.lower() in {".json", ".txt"} and size <= 16 * 1024 * 1024:
                metadata_hashes[relative] = file_sha256(path)
    inventory_hash = hashlib.sha256("\n".join(inventory).encode("utf-8")).hexdigest()
    return {
        "resolved_path": str(root),
        "snapshot_revision": root.name,
        "file_inventory_sha256": inventory_hash,
        "file_count": len(inventory),
        "metadata_file_sha256": metadata_hashes,
        "note": (
            "snapshot_revision is the frozen checkpoint identifier; inventory hash covers "
            "relative names and sizes, while listed lightweight metadata files are byte hashed"
        ),
    }


def validate_resume(existing: list[dict], manifest_by_key: dict[tuple[str, str], dict]) -> set[tuple[str, str]]:
    keys = [(row["item_id"], row["condition"]) for row in existing]
    if len(keys) != len(set(keys)):
        raise ValueError("resume predictions contain duplicate item-condition keys")
    if not set(keys).issubset(manifest_by_key):
        raise ValueError("resume predictions contain keys outside the frozen manifest")
    for row in existing:
        key = (row["item_id"], row["condition"])
        source = manifest_by_key[key]
        for field in (
            "image_sha256", "source_sha256", "verification_question", "correct_semantic",
            "target_semantic", "registered_read_fields",
        ):
            if row.get(field) != source.get(field):
                raise ValueError(f"resume row {key} differs from frozen manifest for {field}")
        reparsed = parse_semantic_answer(
            row.get("answer_raw", ""), source["answer_format"], source["option_order"],
        )
        reread = transcription_fields_match(
            row.get("read_raw", ""), source["registered_read_fields"],
        )
        if row.get("parsed_semantic") != reparsed or bool(row.get("read_match")) != reread:
            raise ValueError(f"resume row {key} has stale derived scoring fields")
    return set(keys)


def write_summary(root: Path, rows: list[dict]) -> None:
    summary = summarize_conditions(rows)
    (root / "summary.json").write_text(
        json.dumps({"conditions": summary}, indent=2) + "\n", encoding="utf-8"
    )
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_schema_version") != SCHEMA_VERSION:
        raise ValueError(f"config protocol_schema_version must equal {SCHEMA_VERSION}")
    expected_hash = str(config.get("expected_manifest_sha256", "")).strip().lower()
    if not re_full_sha256(expected_hash):
        raise ValueError("config must contain the preregistered 64-hex expected_manifest_sha256")
    if bool(config.get("model", {}).get("do_sample", False)):
        raise ValueError("preregistered mechanism controls require greedy decoding (do_sample: false)")

    manifest_path = Path(config["source_manifest"]).resolve()
    actual_hash = file_sha256(manifest_path)
    if actual_hash != expected_hash:
        raise ValueError("current manifest SHA-256 differs from preregistered config")
    manifest = read_jsonl(manifest_path)
    audit = validate_manifest_rows(manifest, check_files=True)
    if audit["items"] != int(config["expected_items"]):
        raise ValueError("manifest item count differs from config")

    build_provenance_path = manifest_path.parent / "build_provenance.json"
    freeze_path = manifest_path.parent / "freeze_record.json"
    build_provenance = json.loads(build_provenance_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if build_provenance.get("status") != "frozen":
        raise ValueError("build provenance is not frozen")
    if build_provenance.get("manifest_sha256") != actual_hash:
        raise ValueError("build provenance does not identify the current manifest")
    if freeze.get("status") != "frozen_before_victim_inference" or freeze.get("manifest_sha256") != actual_hash:
        raise ValueError("freeze record does not identify the current manifest")
    for field in (
        "source_registry_sha256", "builder_sha256", "mechanism_module_sha256",
    ):
        if freeze.get(field) != build_provenance.get(field):
            raise ValueError(f"freeze record and build provenance differ for {field}")
    module_path = Path(__file__).resolve().parents[1] / "cta" / "bridge_mechanism_controls.py"
    if build_provenance.get("mechanism_module_sha256") != file_sha256(module_path):
        raise ValueError("mechanism module changed after manifest freeze; rebuild before inference")

    manifest_by_key = {(row["item_id"], row["condition"]): row for row in manifest}
    expected_keys = set(manifest_by_key)
    output_root = Path(config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = output_root / "predictions.jsonl"
    existing = read_jsonl(predictions_path) if predictions_path.exists() else []
    completed = validate_resume(existing, manifest_by_key)

    seed = int(config.get("seed", 20260901))
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

    # Importing model adapters lazily keeps manifest/provenance validation and
    # ``--help`` usable on CPU-only audit hosts without changing server runs.
    from cta.model import build_model_adapter
    model = build_model_adapter(config["model"])
    questions_sha256 = hashlib.sha256(
        "\n".join(sorted({row["verification_question"] for row in manifest})).encode("utf-8")
    ).hexdigest()
    read_prompts_sha256 = hashlib.sha256(
        "\n".join(sorted({row["read_prompt"] for row in manifest})).encode("utf-8")
    ).hexdigest()
    provenance = {
        "schema_version": f"{SCHEMA_VERSION}/run-provenance",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "manifest": str(manifest_path),
        "manifest_sha256": actual_hash,
        "freeze_record_sha256": file_sha256(freeze_path),
        "build_provenance_sha256": file_sha256(build_provenance_path),
        "runner_sha256": file_sha256(Path(__file__)),
        "mechanism_module_sha256": file_sha256(module_path),
        "questions_sha256": questions_sha256,
        "read_prompts_sha256": read_prompts_sha256,
        "git_head": git_head(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "model": model.provenance(),
        "model_snapshot": model_snapshot_descriptor(config["model"]),
        "items": audit["items"],
        "rows": len(manifest),
        "conditions": list(ALL_CONDITIONS),
        "queries_per_row": 2,
        "decoding": "greedy; no per-item or per-model adaptive search",
        "primary_endpoint": "clean-conditioned read-gated target ASR",
        "seed": seed,
    }
    provenance_path = output_root / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    failures_path = output_root / "failed_queries.jsonl"

    for source in tqdm(
        sorted(manifest, key=lambda row: (row["item_id"], row["condition"])),
        desc="Bridge mechanism controls",
    ):
        key = (source["item_id"], source["condition"])
        if key in completed:
            continue
        started = time.time()
        answer_raw = None
        try:
            answer_raw = model.infer(source["image_path"], source["verification_question"])
            read_raw = model.infer(source["image_path"], source["read_prompt"])
        except Exception as exc:
            append_jsonl(failures_path, {
                "item_id": source["item_id"],
                "condition": source["condition"],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "answer_query_completed": answer_raw is not None,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            })
            raise
        parsed = parse_semantic_answer(
            answer_raw, source["answer_format"], source["option_order"],
        )
        read_match = transcription_fields_match(read_raw, source["registered_read_fields"])
        row = {
            **source,
            "answer_raw": answer_raw,
            "parsed_semantic": parsed,
            "answer_correct": parsed == source["correct_semantic"],
            "target_match": parsed == source["target_semantic"],
            "read_raw": read_raw,
            "read_match": read_match,
            "read_gated_target_success": (
                parsed == source["target_semantic"] and read_match
            ),
            "latency_s": round(time.time() - started, 4),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "inference_metadata": (
                model.inference_metadata() if hasattr(model, "inference_metadata") else {}
            ),
        }
        append_jsonl(predictions_path, row)
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


def re_full_sha256(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", value))


if __name__ == "__main__":
    main()
