#!/usr/bin/env python3
"""Measure the answer prior when the registered record is absent.

This is a diagnostic, not a clean-accuracy condition: the question refers to a
record that is intentionally missing from the unmodified source image.  It
tests whether question wording alone already elicits the false-record target.
"""

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

from cta.contraledger import parse_answer, source_prior_items, summarize_source_prior
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
    items = source_prior_items(read_jsonl(manifest_path))
    if len(items) != int(config["expected_items"]):
        raise ValueError("source-prior item count does not match config")

    output_root = Path(str(Path(config["output_root"]).resolve()) + "_source_prior")
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "predictions.jsonl"
    existing = read_jsonl(output_path)
    completed = {str(row["item_id"]) for row in existing}
    expected = {str(row["item_id"]) for row in items}
    if not completed.issubset(expected) or len(completed) != len(existing):
        raise ValueError("source-prior log has duplicate or unregistered items")

    model = build_model_adapter(config["model"])
    provenance = {
        "schema_version": "cta/contraledger-source-prior-run-v1",
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
        "items": len(items),
        "diagnostic_boundary": "Question-prior audit on source images without the registered record; no accuracy claim.",
    }
    provenance_path = output_root / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    for source in tqdm(items, desc="ContraLedger source-prior"):
        if source["item_id"] in completed:
            continue
        raw = model.infer(source["source_path"], source["question"])
        row = {
            **source,
            "prior_raw": raw,
            "prior_parsed": parse_answer(raw, source["option_order"]),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "inference_metadata": model.inference_metadata() if hasattr(model, "inference_metadata") else {},
        }
        append_jsonl(output_path, row)
        existing.append(row)
        completed.add(source["item_id"])

    summary = summarize_source_prior(existing)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    provenance["completed_rows"] = len(existing)
    provenance["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["model"] = model.provenance()
    provenance["status"] = "complete" if completed == expected else "incomplete"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    if provenance["status"] != "complete":
        raise RuntimeError(f"source-prior run incomplete: {len(completed)}/{len(expected)}")


if __name__ == "__main__":
    main()
