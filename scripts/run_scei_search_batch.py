#!/usr/bin/env python3
"""Run a frozen, resumable bounded SCEI-Search batch."""

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
from cta.scei_adaptive import adaptive_scei_events
from cta.scei_batch import (
    freeze_selection,
    load_jsonl,
    safe_item_slug,
    summarize_terminal_rows,
    terminal_row,
)


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


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def next_attempt_dir(item_root: Path) -> Path:
    item_root.mkdir(parents=True, exist_ok=True)
    indexes = []
    for path in item_root.glob("attempt_*"):
        try:
            indexes.append(int(path.name.split("_")[-1]))
        except ValueError:
            continue
    attempt = item_root / f"attempt_{max(indexes, default=0) + 1:03d}"
    attempt.mkdir()
    return attempt


def seed_everything(seed: int) -> None:
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
    offset = int(config.get("offset", 48))
    max_rounds = int(config.get("max_rounds", 2))
    renderer = str(config.get("renderer", "scene"))
    strict_read_gate = bool(config.get("strict_read_gate", True))
    planner_attempts = int(config.get("max_planner_attempts", 3))
    counterfactual_families = [str(value) for value in config.get("counterfactual_families", [])]
    if not counterfactual_families:
        raise ValueError("counterfactual_families must be explicitly frozen in the config")
    if max_rounds != 2:
        raise ValueError("this frozen protocol requires max_rounds: 2")
    if not strict_read_gate:
        raise ValueError("this frozen protocol requires strict_read_gate: true")
    if renderer != "scene":
        raise ValueError("this frozen protocol requires renderer: scene")

    output_root.mkdir(parents=True, exist_ok=True)
    selection_path = output_root / "selection_manifest.json"
    lock_path = output_root / "run_lock.json"
    config_hash = file_sha256(config_path)
    source_hash = file_sha256(source_manifest)
    if selection_path.exists() != lock_path.exists():
        raise RuntimeError("selection manifest and run lock must either both exist or both be absent")
    if not selection_path.exists():
        selection = freeze_selection(
            source_manifest,
            seed=seed,
            offset=offset,
            limit=expected_items,
            families=counterfactual_families,
        )
        write_json(selection_path, selection)
        selection_hash = file_sha256(selection_path)
        lock = {
            "schema_version": "cta/scei-search-run-lock-v1",
            "status": "frozen-before-model-load",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path),
            "config_sha256": config_hash,
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": source_hash,
            "selection_manifest": str(selection_path),
            "selection_manifest_sha256": selection_hash,
            "selection_rule": (
                "exclude the exact SCEI development prefix, then allocate fixed family quotas using only "
                "scene-label compatibility and deterministic SHA-256 order; no victim filtering"
            ),
            "selection_seed": seed,
            "selection_offset": offset,
            "expected_items": expected_items,
            "maximum_rounds": max_rounds,
            "strict_read_gate": strict_read_gate,
            "renderer": renderer,
            "counterfactual_families": counterfactual_families,
            "planner_model": config["planner_model"],
            "victim_model": config["victim_model"],
            "git_head_at_freeze": git_head(),
            "hostname_at_freeze": platform.node(),
        }
        write_json(lock_path, lock)
    else:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        checks = {
            "config_sha256": config_hash,
            "source_manifest_sha256": source_hash,
            "selection_manifest_sha256": file_sha256(selection_path),
            "selection_seed": seed,
            "selection_offset": offset,
            "expected_items": expected_items,
            "maximum_rounds": max_rounds,
            "strict_read_gate": strict_read_gate,
            "renderer": renderer,
            "counterfactual_families": counterfactual_families,
        }
        mismatched = {key: (lock.get(key), value) for key, value in checks.items() if lock.get(key) != value}
        if mismatched:
            raise RuntimeError(f"frozen run lock mismatch: {mismatched}")
    if not isinstance(selection, list) or len(selection) != expected_items:
        raise RuntimeError("selection manifest length does not equal expected_items")

    results_path = output_root / "sample_results.jsonl"
    existing = load_jsonl(results_path)
    completed = {str(row["item_id"]): row for row in existing}
    if len(completed) != len(existing):
        raise RuntimeError("sample_results.jsonl contains duplicate item ids")
    selected_ids = {str(row["item_id"]) for row in selection}
    if not set(completed).issubset(selected_ids):
        raise RuntimeError("sample_results.jsonl contains ids absent from the frozen selection")

    # Freeze and validate all selection evidence before either model is loaded.
    for row in selection:
        source = Path(row["source_path"])
        if not source.is_file() or file_sha256(source) != row["source_sha256"]:
            raise RuntimeError(f"source evidence changed: {row['item_id']}")
    seed_everything(seed)
    planner_cfg = config["planner_model"]
    victim_cfg = config["victim_model"]
    planner = build_model_adapter(planner_cfg)
    victim = planner if planner_cfg == victim_cfg else build_model_adapter(victim_cfg)
    provenance_path = output_root / "provenance.json"
    provenance = {
        "schema_version": "cta/scei-search-provenance-v1",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": file_sha256(selection_path),
        "run_lock": str(lock_path),
        "run_lock_sha256": file_sha256(lock_path),
        "git_head": git_head(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "planner": planner.provenance(),
        "victim": victim.provenance(),
        "query_policy": (
            "one clean answer query; per attack round one answer plus one independent exact-transcription query; "
            "round two may use round-one feedback; stop on strict success or K=2"
        ),
        "success_definition": (
            "clean answer rejects the false claim, attacked answer accepts it, and the complete registered "
            "measurement including uncertainty is transcribed exactly"
        ),
        "reporting_population": "Success@K is clean-conditioned; clean errors remain in the selected-n accounting",
    }
    write_json(provenance_path, provenance)

    for index, item in enumerate(selection, 1):
        item_id = str(item["item_id"])
        if item_id in completed:
            continue
        item_root = output_root / "items" / safe_item_slug(item_id)
        # If inference finished before a process interruption, recover its terminal row without re-querying.
        recovered = None
        for candidate in sorted(item_root.glob("attempt_*"), reverse=True):
            if (candidate / "summary.json").is_file() and (candidate / "events.jsonl").is_file():
                summary = json.loads((candidate / "summary.json").read_text(encoding="utf-8"))
                recovered = terminal_row(item, candidate, summary, load_jsonl(candidate / "events.jsonl"))
                break
        if recovered is not None:
            append_jsonl(results_path, recovered)
            completed[item_id] = recovered
            continue
        attempt_dir = next_attempt_dir(item_root)
        try:
            for event in adaptive_scei_events(
                item["source_path"],
                item["target_label"],
                planner,
                victim,
                attempt_dir,
                visible_labels=item["visible_labels"],
                counterfactual_family=item["family"],
                max_rounds=max_rounds,
                renderer_mode=renderer,
                strict_read_gate=strict_read_gate,
                max_planner_attempts=planner_attempts,
            ):
                print(json.dumps({
                    "item": f"{index}/{expected_items}",
                    "item_id": item_id,
                    "stage": event["stage"],
                    "round": event["round"],
                    "parsed_semantic": event["parsed_semantic"],
                    "exact_read_match": event.get("exact_read_match"),
                    "success": event["success"],
                    "feedback_class": event["feedback_class"],
                }, ensure_ascii=False), flush=True)
        except Exception as exc:
            write_json(attempt_dir / "failure.json", {
                "item_id": item_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })
            raise
        summary = json.loads((attempt_dir / "summary.json").read_text(encoding="utf-8"))
        row = terminal_row(item, attempt_dir, summary, load_jsonl(attempt_dir / "events.jsonl"))
        append_jsonl(results_path, row)
        completed[item_id] = row
        partial = summarize_terminal_rows(
            list(completed.values()), expected_items=expected_items, max_rounds=max_rounds
        )
        write_json(output_root / "analysis_partial.json", partial)

    terminal = load_jsonl(results_path)
    analysis = summarize_terminal_rows(terminal, expected_items=expected_items, max_rounds=max_rounds)
    if analysis["status"] != "complete":
        raise RuntimeError(f"batch incomplete: {len(terminal)}/{expected_items}")
    analysis.update({
        "selection_manifest_sha256": file_sha256(selection_path),
        "sample_results_sha256": file_sha256(results_path),
        "analysis_scope": (
            "development-disjoint from SCEI offsets 0-47, but drawn from the existing COCO n=300 pool; "
            "not a new public benchmark and not a cross-model SOTA comparison"
        ),
    })
    write_json(output_root / "analysis.json", analysis)
    provenance["status"] = "complete"
    provenance["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["terminal_items"] = len(terminal)
    provenance["sample_results_sha256"] = file_sha256(results_path)
    provenance["analysis_sha256"] = file_sha256(output_root / "analysis.json")
    write_json(provenance_path, provenance)
    print(json.dumps(analysis, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
