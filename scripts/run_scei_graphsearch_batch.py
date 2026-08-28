#!/usr/bin/env python3
"""Run a frozen, resumable SCEI-GraphSearch batch."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.model import build_model_adapter
from cta.question_bench import file_sha256
from cta.scei_batch import freeze_selection, load_jsonl, safe_item_slug
from cta.scei_graphsearch import graphsearch_scei_events


def _git_head() -> str:
    try:
        root = Path(__file__).resolve().parents[1]
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "not-a-git-checkout"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _seed_everything(seed: int) -> None:
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


def _summarize(rows: list[dict], expected_items: int, max_rounds: int) -> dict:
    successes = [row for row in rows if row.get("success")]
    success_rounds = [int(row["first_success_round"]) for row in successes]
    query_counts = [int(row["victim_query_count"]) for row in rows]
    by_family: dict[str, list[dict]] = {}
    for row in rows:
        by_family.setdefault(str(row["selection_family"]), []).append(row)
    return {
        "schema_version": "cta/scei-graphsearch-batch-summary-v1",
        "expected_items": expected_items,
        "completed_items": len(rows),
        "maximum_attack_rounds": max_rounds,
        "strict_successes": len(successes),
        "strict_success_rate_selected": len(successes) / len(rows) if rows else None,
        "mean_rounds_to_success": sum(success_rounds) / len(success_rounds) if success_rounds else None,
        "mean_victim_queries_selected": sum(query_counts) / len(query_counts) if query_counts else None,
        "by_selection_family": [
            {
                "family": family,
                "n": len(group),
                "strict_successes": sum(bool(row.get("success")) for row in group),
                "strict_success_rate": sum(bool(row.get("success")) for row in group) / len(group),
            }
            for family, group in sorted(by_family.items())
        ],
        "success_definition": "clean-gated target flip plus exact transcription within the fixed round budget",
        "reporting_note": "selected-n denominator; all clean failures and exhausted searches remain counted",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_manifest = Path(config["source_manifest"]).resolve()
    output_root = Path(config["output_root"]).resolve()
    expected_items = int(config["expected_items"])
    seed = int(config.get("seed", 20260827))
    record_seed = int(config.get("record_seed", 20260828))
    offset = int(config.get("offset", 48))
    max_rounds = int(config.get("max_rounds", 2))
    max_families = int(config.get("max_families", 4))
    renderer = str(config.get("renderer", "scene"))
    strict_read_gate = bool(config.get("strict_read_gate", True))
    selection_families = [str(value) for value in config["selection_families"]]
    output_root.mkdir(parents=True, exist_ok=True)

    selection_path = output_root / "selection_manifest.json"
    lock_path = output_root / "run_lock.json"
    if selection_path.exists() != lock_path.exists():
        raise RuntimeError("selection manifest and run lock must either both exist or both be absent")
    if not selection_path.exists():
        selection = freeze_selection(
            source_manifest,
            seed=seed,
            offset=offset,
            limit=expected_items,
            families=selection_families,
        )
        _write_json(selection_path, selection)
        _write_json(lock_path, {
            "schema_version": "cta/scei-graphsearch-run-lock-v1",
            "status": "frozen-before-model-load",
            "config_path": str(config_path),
            "config_sha256": file_sha256(config_path),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": file_sha256(source_manifest),
            "selection_manifest": str(selection_path),
            "selection_manifest_sha256": file_sha256(selection_path),
            "expected_items": expected_items,
            "seed": seed,
            "record_seed": record_seed,
            "offset": offset,
            "max_rounds": max_rounds,
            "max_families": max_families,
            "renderer": renderer,
            "strict_read_gate": strict_read_gate,
            "selection_families": selection_families,
            "victim_model": config["victim_model"],
            "git_head_at_freeze": _git_head(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })
    else:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        checks = {
            "config_sha256": file_sha256(config_path),
            "source_manifest_sha256": file_sha256(source_manifest),
            "selection_manifest_sha256": file_sha256(selection_path),
            "expected_items": expected_items,
            "seed": seed,
            "record_seed": record_seed,
            "offset": offset,
            "max_rounds": max_rounds,
            "max_families": max_families,
            "renderer": renderer,
            "strict_read_gate": strict_read_gate,
            "selection_families": selection_families,
        }
        mismatch = {key: (lock.get(key), value) for key, value in checks.items() if lock.get(key) != value}
        if mismatch:
            raise RuntimeError(f"frozen run lock mismatch: {mismatch}")
    if len(selection) != expected_items:
        raise RuntimeError("selection manifest length does not equal expected_items")
    for row in selection:
        source = Path(row["source_path"])
        if not source.is_file() or file_sha256(source) != row["source_sha256"]:
            raise RuntimeError(f"source evidence changed: {row['item_id']}")

    _seed_everything(seed)
    victim = build_model_adapter(config["victim_model"])
    _write_json(output_root / "provenance.json", {
        "schema_version": "cta/scei-graphsearch-provenance-v1",
        "status": "running",
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "run_lock_sha256": file_sha256(lock_path),
        "selection_manifest_sha256": file_sha256(selection_path),
        "git_head": _git_head(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "victim": victim.provenance(),
        "query_policy": "per new semantic arm one clean gate; per attack round one answer and one exact-read query",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    })

    results_path = output_root / "sample_results.jsonl"
    existing = load_jsonl(results_path)
    completed = {str(row["item_id"]): row for row in existing}
    if len(completed) != len(existing):
        raise RuntimeError("duplicate item ids in sample_results.jsonl")
    for index, item in enumerate(selection, 1):
        item_id = str(item["item_id"])
        if item_id in completed:
            continue
        item_root = output_root / "items" / safe_item_slug(item_id)
        item_root.mkdir(parents=True, exist_ok=True)
        attempt = item_root / "attempt_001"
        if attempt.exists() and not (attempt / "summary.json").is_file():
            raise RuntimeError(f"incomplete attempt requires manual audit: {attempt}")
        attempt.mkdir(exist_ok=True)
        if not (attempt / "summary.json").is_file():
            for event in graphsearch_scei_events(
                item["source_path"],
                item["target_label"],
                victim,
                attempt,
                visible_labels=item["visible_labels"],
                record_seed=record_seed,
                max_rounds=max_rounds,
                max_families=max_families,
                renderer_mode=renderer,
                strict_read_gate=strict_read_gate,
            ):
                print(json.dumps({
                    "item": f"{index}/{expected_items}",
                    "item_id": item_id,
                    "stage": event["stage"],
                    "round": event.get("attack_round"),
                    "arm_id": event.get("arm_id"),
                    "feedback": event["feedback_class"],
                    "success": event["success"],
                }, ensure_ascii=False), flush=True)
        summary = json.loads((attempt / "summary.json").read_text(encoding="utf-8"))
        row = {
            "schema_version": "cta/scei-graphsearch-sample-result-v1",
            "item_id": item_id,
            "selection_family": item["family"],
            "target_label": item["target_label"],
            "source_sha256": item["source_sha256"],
            "attempt_dir": str(attempt),
            "success": bool(summary["success"]),
            "first_success_round": summary["first_success_round"],
            "attack_rounds_used": summary["attack_rounds_used"],
            "semantic_arms_clean_gated": summary["semantic_arms_clean_gated"],
            "victim_query_count": summary["victim_query_count"],
            "candidate_bank_sha256": summary["candidate_bank_sha256"],
        }
        _append_jsonl(results_path, row)
        completed[item_id] = row
        _write_json(output_root / "summary.json", _summarize(list(completed.values()), expected_items, max_rounds))

    provenance = json.loads((output_root / "provenance.json").read_text(encoding="utf-8"))
    provenance.update(status="complete", finished_at_utc=datetime.now(timezone.utc).isoformat())
    _write_json(output_root / "provenance.json", provenance)


if __name__ == "__main__":
    main()
