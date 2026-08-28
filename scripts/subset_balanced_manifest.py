#!/usr/bin/env python3
"""Freeze a stratified item subset of an existing balanced-v1 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import file_sha256


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--selection-seed", type=int, default=20260828)
    args = parser.parse_args()
    source = args.source_manifest.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    rows = read_jsonl(source)
    by_cell: dict[str, set[str]] = {}
    for row in rows:
        by_cell.setdefault(row["counterbalance_cell"], set()).add(row["item_id"])
    cells = sorted(by_cell)
    if args.limit % len(cells):
        raise ValueError(f"limit must be divisible by {len(cells)}")
    selected = set()
    for cell in cells:
        ranked = sorted(
            by_cell[cell],
            key=lambda item_id: hashlib.sha256(f"{args.selection_seed}:{cell}:{item_id}".encode()).hexdigest(),
        )
        selected.update(ranked[:args.limit // len(cells)])
    subset = [row for row in rows if row["item_id"] in selected]
    conditions = sorted({row["condition"] for row in rows})
    expected = {(item_id, condition) for item_id in selected for condition in conditions}
    actual = {(row["item_id"], row["condition"]) for row in subset}
    if len(selected) != args.limit or actual != expected:
        raise ValueError("subset coverage audit failed")
    output.mkdir(parents=True)
    manifest = output / "render_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in subset),
        encoding="utf-8",
    )
    provenance = {
        "schema_version": "cta/balanced-subset-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source),
        "source_manifest_sha256": file_sha256(source),
        "selection_seed": args.selection_seed,
        "items": len(selected), "conditions": conditions, "rows": len(subset),
        "cell_counts": {
            cell: len({row["item_id"] for row in subset if row["counterbalance_cell"] == cell})
            for cell in cells
        },
        "selected_item_sha256": hashlib.sha256("\n".join(sorted(selected)).encode()).hexdigest(),
        "manifest_sha256": file_sha256(manifest),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"items": len(selected), "rows": len(subset), "manifest": str(manifest)}))


if __name__ == "__main__":
    main()
