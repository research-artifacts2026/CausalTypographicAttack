#!/usr/bin/env python3
"""Audit and recompute SCEI-Search batch metrics from terminal evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import file_sha256
from cta.scei_batch import load_jsonl, summarize_terminal_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    root = args.run_root.resolve()
    lock = json.loads((root / "run_lock.json").read_text(encoding="utf-8"))
    selection_path = root / "selection_manifest.json"
    results_path = root / "sample_results.jsonl"
    if file_sha256(selection_path) != lock["selection_manifest_sha256"]:
        raise RuntimeError("selection manifest hash differs from frozen run lock")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = {str(row["item_id"]): row for row in selection}
    rows = load_jsonl(results_path)
    if len(rows) != len({str(row["item_id"]) for row in rows}):
        raise RuntimeError("duplicate terminal item ids")
    if not {str(row["item_id"]) for row in rows}.issubset(selected):
        raise RuntimeError("terminal results include an unselected item")
    for row in rows:
        item = selected[str(row["item_id"])]
        if row["source_sha256"] != item["source_sha256"]:
            raise RuntimeError(f"source hash mismatch: {row['item_id']}")
        for path_key, hash_key in (
            ("protocol_path", "protocol_sha256"),
            ("events_path", "events_sha256"),
            ("summary_path", "summary_sha256"),
        ):
            path = Path(row[path_key])
            if not path.is_file() or file_sha256(path) != row[hash_key]:
                raise RuntimeError(f"evidence hash mismatch: {row['item_id']} {path_key}")
        events = load_jsonl(Path(row["events_path"]))
        for event in events:
            if event.get("stage") != "attack":
                continue
            expected_success = bool(event.get("target_match")) and bool(event.get("exact_read_match"))
            if bool(event.get("success")) != expected_success:
                raise RuntimeError(f"strict-gate mismatch: {row['item_id']} round {event.get('round')}")
            for path_key, hash_key in (
                ("image_path", "image_sha256"),
                ("mask_path", "mask_sha256"),
            ):
                path = Path(str(event[path_key]))
                if not path.is_file() or file_sha256(path) != event[hash_key]:
                    raise RuntimeError(
                        f"rendered evidence hash mismatch: {row['item_id']} round {event.get('round')} {path_key}"
                    )
    expected = int(lock["expected_items"])
    analysis = summarize_terminal_rows(
        rows, expected_items=expected, max_rounds=int(lock["maximum_rounds"])
    )
    if analysis["status"] != "complete" and not args.allow_incomplete:
        raise RuntimeError(f"batch incomplete: {len(rows)}/{expected}")
    analysis.update({
        "selection_manifest_sha256": file_sha256(selection_path),
        "sample_results_sha256": file_sha256(results_path),
        "audited": True,
    })
    output = root / ("analysis.json" if analysis["status"] == "complete" else "analysis_partial.json")
    output.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
