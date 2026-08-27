#!/usr/bin/env python3
"""Independently audit a frozen SCEI five-condition evaluation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


CONDITIONS = {"clean_false", "clean_true", "flat_false", "scene_false", "scene_true"}
READ_CONDITIONS = {"flat_false", "scene_false", "scene_true"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-items", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors = []
    grouped = defaultdict(dict)
    for row in rows:
        key = (str(row["item_id"]), str(row["condition"]))
        if key[1] in grouped[key[0]]:
            errors.append(f"duplicate:{key[0]}:{key[1]}")
        grouped[key[0]][key[1]] = row
        image_path = Path(row["image_path"])
        if not image_path.is_file() or sha256(image_path) != row["image_sha256"]:
            errors.append(f"image_hash:{key[0]}:{key[1]}")
        if row.get("mask_path"):
            mask_path = Path(row["mask_path"])
            if not mask_path.is_file() or sha256(mask_path) != row["mask_sha256"]:
                errors.append(f"mask_hash:{key[0]}:{key[1]}")
        if (key[1] in READ_CONDITIONS) != bool(row.get("requires_read")):
            errors.append(f"read_flag:{key[0]}:{key[1]}")

    if len(grouped) != args.expected_items:
        errors.append(f"item_count:{len(grouped)}")
    for item_id, values in grouped.items():
        if set(values) != CONDITIONS:
            errors.append(f"condition_set:{item_id}")
            continue
        if values["clean_false"]["image_sha256"] != values["clean_true"]["image_sha256"]:
            errors.append(f"clean_pair:{item_id}")
        if values["scene_false"]["mask_sha256"] != values["scene_true"]["mask_sha256"]:
            errors.append(f"scene_mask_pair:{item_id}")
        if values["scene_false"]["carrier_quad"] != values["scene_true"]["carrier_quad"]:
            errors.append(f"scene_geometry_pair:{item_id}")
        cells = {row["counterbalance_cell"] for row in values.values()}
        if len(cells) != 1:
            errors.append(f"answer_cell_pair:{item_id}")
        false_questions = {
            values[name]["verification_question"]
            for name in ("clean_false", "flat_false", "scene_false")
        }
        if len(false_questions) != 1:
            errors.append(f"false_question_pair:{item_id}")
        if any(field in row for row in values.values() for field in ("answer_raw", "read_raw", "parsed_semantic")):
            errors.append(f"victim_output_leak:{item_id}")

    item_rows = [values["clean_false"] for values in grouped.values() if "clean_false" in values]
    audit = {
        "schema_version": "cta/scei-image-eval-audit-v1",
        "status": "passed" if not errors else "failed",
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "expected_items": args.expected_items,
        "items": len(grouped),
        "rows": len(rows),
        "expected_rows": args.expected_items * len(CONDITIONS),
        "answer_cell_counts": dict(sorted(Counter(row["counterbalance_cell"] for row in item_rows).items())),
        "family_counts": dict(sorted(Counter(row["family"] for row in item_rows).items())),
        "error_count": len(errors),
        "errors": errors,
        "boundary": "manifest contains no victim outputs and was audited before inference",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

