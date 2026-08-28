#!/usr/bin/env python3
"""Build a renderer-matched Bridge benchmark from a frozen balanced manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.bridge_renderer import render_bridge_carrier
from cta.question_bench import file_sha256


CONDITIONS = ("no_attack", "bridge_flat", "bridge_scene")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def choose_items(rows: list[dict], limit: int, seed: int) -> set[str]:
    by_cell: dict[str, set[str]] = {}
    for row in rows:
        by_cell.setdefault(row["counterbalance_cell"], set()).add(row["item_id"])
    cells = sorted(by_cell)
    if limit % len(cells):
        raise ValueError(f"limit {limit} must be divisible by {len(cells)} cells")
    selected = set()
    for cell in cells:
        ranked = sorted(
            by_cell[cell],
            key=lambda item_id: hashlib.sha256(f"{seed}:{cell}:{item_id}".encode()).hexdigest(),
        )
        selected.update(ranked[:limit // len(cells)])
    if len(selected) != limit:
        raise ValueError("stratified selection failed")
    return selected


def git_head() -> str:
    root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "not-a-git-checkout"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--selection-seed", type=int, default=20260828)
    args = parser.parse_args()
    source_manifest = args.source_manifest.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing benchmark: {output_root}")
    output_root.mkdir(parents=True)
    source_rows = read_jsonl(source_manifest)
    selected = choose_items(source_rows, args.limit, args.selection_seed)
    indexed = {(row["item_id"], row["condition"]): row for row in source_rows}
    output_rows = []
    content_audit = []
    for item_id in sorted(selected):
        clean = dict(indexed[(item_id, "no_attack")])
        bridge = indexed[(item_id, "causal_bridge")]
        parts = bridge["overlay_text"].split(" | ")
        if len(parts) != 3 or parts[0] != bridge["attack_claim"] or parts[2] != "STATUS VERIFIED":
            raise ValueError(f"unexpected frozen Bridge text structure for {item_id}")
        clean["condition"] = "no_attack"
        clean["renderer"] = "frozen-balanced-v1-clean"
        output_rows.append(clean)
        render_hashes = {}
        for mode, condition in (("flat", "bridge_flat"), ("scene", "bridge_scene")):
            result = render_bridge_carrier(
                source=bridge["source_path"],
                output=output_root / "images" / condition / f"{item_id}.jpg",
                item_id=item_id,
                bbox=tuple(bridge["bbox"]),
                mode=mode,
                title="WORLD MODEL AUDIT",
                proposition=parts[0],
                conclusion=parts[1],
                status=parts[2],
            )
            row = dict(bridge)
            row.update(result)
            row["condition"] = condition
            row["placement"] = bridge["placement"]
            row["matched_source_condition"] = "causal_bridge"
            row["rendered_components"] = {
                "title": "WORLD MODEL AUDIT", "proposition": parts[0],
                "conclusion": parts[1], "status": parts[2],
            }
            output_rows.append(row)
            render_hashes[condition] = result["image_sha256"]
        content_audit.append({
            "item_id": item_id,
            "overlay_text_sha256": hashlib.sha256(bridge["overlay_text"].encode()).hexdigest(),
            "flat_image_sha256": render_hashes["bridge_flat"],
            "scene_image_sha256": render_hashes["bridge_scene"],
        })
    output_rows.sort(key=lambda row: (row["item_id"], CONDITIONS.index(row["condition"])))
    manifest = output_root / "render_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    provenance = {
        "schema_version": "cta/matched-bridge-renderer-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "hostname": platform.node(),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": file_sha256(source_manifest),
        "selection_seed": args.selection_seed,
        "items": len(selected),
        "rows": len(output_rows),
        "counterbalance_counts": {
            cell: len({row["item_id"] for row in output_rows if row["counterbalance_cell"] == cell})
            for cell in sorted({row["counterbalance_cell"] for row in output_rows})
        },
        "content_lock": "identical proposition, conclusion, status, title, bbox, and font geometry across flat/scene",
        "scene_branch_boundary": "deterministic synthetic perspective/tone integration; not camera capture and not AI generation",
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "content_audit": content_audit,
    }
    (output_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"items": len(selected), "rows": len(output_rows), "manifest": str(manifest)}))


if __name__ == "__main__":
    main()
