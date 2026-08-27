#!/usr/bin/env python3
"""Build a provenance-locked three-image distributed SCEI subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import file_sha256
from cta.scei_batch import read_json_records, select_family_balanced
from cta.scei_distributed import compile_distributed_ledger, render_distributed_ledger


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


def canonical_rows(rows: list[dict]) -> list[dict]:
    unique = {}
    for row in rows:
        item_id = str(row.get("sample_id", row.get("item_id", ""))).strip()
        if not item_id:
            raise ValueError("source row lacks item id")
        unique.setdefault(item_id, dict(row))
    return list(unique.values())


def select_triplets(rows: list[dict], *, seed: int, expected: int, excluded: set[str]) -> list[dict]:
    groups = defaultdict(list)
    for row in canonical_rows(rows):
        item_id = str(row.get("sample_id", row.get("item_id")))
        if item_id in excluded:
            continue
        label = str(row["target_label"]).strip().lower()
        groups[label].append(row)
    candidates = []
    for label, label_rows in groups.items():
        label_rows.sort(key=lambda row: hashlib.sha256(
            f"{seed}:{row.get('sample_id', row.get('item_id'))}:scei-distributed-v1".encode("utf-8")
        ).hexdigest())
        for chunk_index in range(len(label_rows) // 3):
            chunk = label_rows[chunk_index * 3:(chunk_index + 1) * 3]
            priority = hashlib.sha256(f"{seed}:{label}:{chunk_index}:triplet".encode("utf-8")).hexdigest()
            candidates.append((chunk_index, priority, label, chunk))
    # Prefer one triplet per object class before taking another from a repeated class.
    candidates.sort(key=lambda value: (value[0], value[1], value[2]))
    selected = candidates[:expected]
    if len(selected) != expected:
        raise ValueError(f"requested {expected} triplets, found {len(selected)}")
    output = []
    for index, (_, _, label, chunk) in enumerate(selected):
        source_rows = []
        for row in chunk:
            item_id = str(row.get("sample_id", row.get("item_id")))
            source = Path(str(row.get("source_path", row.get("image_path", "")))).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            actual_hash = file_sha256(source)
            expected_hash = str(row.get("source_sha256", actual_hash))
            if expected_hash != actual_hash:
                raise ValueError(f"{item_id}: source SHA-256 mismatch")
            source_rows.append({
                "item_id": item_id,
                "source_path": str(source),
                "source_sha256": actual_hash,
            })
        output.append({
            "triplet_id": f"distributed-{index:03d}-{label.replace(' ', '_')}",
            "target_label": label,
            "sources": source_rows,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_manifest = Path(config["source_manifest"]).resolve()
    exclusion_manifest = Path(config["exclude_selection_manifest"]).resolve()
    output_root = Path(config["output_root"]).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite frozen distributed dataset: {output_root}")
    excluded_selection = json.loads(exclusion_manifest.read_text(encoding="utf-8"))
    excluded = {str(row["item_id"]) for row in excluded_selection}
    development = select_family_balanced(
        read_json_records(source_manifest),
        seed=int(config.get("seed", 20260827)),
        offset=0,
        limit=int(config.get("development_exclusion_count", 48)),
    )
    excluded.update(str(row.get("sample_id", row.get("item_id"))) for row in development)
    selection = select_triplets(
        read_json_records(source_manifest),
        seed=int(config.get("seed", 20260827)),
        expected=int(config["expected_triplets"]),
        excluded=excluded,
    )
    output_root.mkdir(parents=True)
    selection_path = output_root / "selection_manifest.json"
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = []
    for item in selection:
        record = compile_distributed_ledger(item["target_label"])
        artifact = render_distributed_ledger(
            [row["source_path"] for row in item["sources"]],
            record,
            output_root / "items" / item["triplet_id"],
            item["triplet_id"],
        )
        rows.append({**item, **artifact})
    manifest_path = output_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": "cta/scei-distributed-build-v1",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "hostname": platform.node(),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": file_sha256(source_manifest),
        "excluded_selection_manifest": str(exclusion_manifest),
        "excluded_selection_manifest_sha256": file_sha256(exclusion_manifest),
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": file_sha256(selection_path),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "triplets": len(rows),
        "native_panel_slots": len(rows) * 3,
        "rendered_panel_images_false_and_corrected": len(rows) * 6,
        "rendered_panel_masks_false_and_corrected": len(rows) * 6,
        "triptych_images_false_and_corrected": len(rows) * 2,
        "object_counts": dict(sorted(Counter(row["target_label"] for row in rows).items())),
        "evaluation_boundary": (
            "this builder creates native three-file records and convenience triptychs; no model result is produced, "
            "and a triptych result cannot be described as native multi-image evaluation"
        ),
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
