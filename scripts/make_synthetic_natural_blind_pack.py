#!/usr/bin/env python3
"""Prepare, but never self-fill, a blinded three-person naturalness pack."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_human_eval_pack import RATING_COLUMNS, opaque_id, write_csv, write_html_form


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--annotators", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    registry_path = args.registry.resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    output_root = args.output_root.resolve()
    images_root = output_root / "images"
    images_root.mkdir(parents=True, exist_ok=False)
    repo_root = Path(__file__).resolve().parents[1]
    public_items = []
    private_rows = []
    for item in registry["items"]:
        opaque = opaque_id(args.seed, item["item_id"])
        source = (repo_root / item["image_path"]).resolve()
        destination = images_root / f"{opaque}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        public_items.append({
            "item_id": opaque,
            "image": f"images/{destination.name}",
            "claim": item["registered_read_text"],
        })
        private_rows.append({
            "item_id": opaque,
            "sample_id": item["item_id"],
            "method": "synthetic_natural",
            "source_image": str(source),
            "claim": item["registered_read_text"],
        })
    fields = ["row_id", "item_id", "image", "claim", *RATING_COLUMNS, "comments"]
    for index in range(args.annotators):
        rng = random.Random(args.seed + index)
        order = list(public_items)
        rng.shuffle(order)
        duplicate = rng.choice(order)
        order.append({**duplicate, "duplicate": True})
        rng.shuffle(order)
        rows = [{
            "row_id": f"A{index + 1}-{row_index:04d}",
            "item_id": item["item_id"],
            "image": item["image"],
            "claim": item["claim"],
            **{column: "" for column in RATING_COLUMNS},
            "comments": "",
        } for row_index, item in enumerate(order, start=1)]
        write_csv(output_root / "assignments" / f"annotator_{index + 1}.csv", rows, fields)
        write_html_form(output_root / "forms" / f"annotator_{index + 1}.html", rows, f"annotator_{index + 1}")
    write_csv(
        output_root / "private_method_key.csv",
        private_rows,
        ["item_id", "sample_id", "method", "source_image", "claim"],
    )
    (output_root / "README.md").write_text(
        "# Synthetic natural-render blind pack\n\n"
        "Status: awaiting three independent human response files. These AI-edited images are synthetic, not real physical captures. "
        "Keep `private_method_key.csv` hidden until all responses are locked. A model may be analyzed only with "
        "`--evaluator-kind model --evaluator-model <name>` and must never be reported as a human annotator.\n",
        encoding="utf-8",
    )
    (output_root / "provenance.json").write_text(json.dumps({
        "schema_version": "cta/synthetic-natural-blind-pack-v1",
        "status": "awaiting independent human annotations",
        "annotators_required": args.annotators,
        "items": len(public_items),
        "hidden_repeats_per_annotator": 1,
        "seed": args.seed,
        "evidence_label": "synthetic natural-render; not real physical capture",
    }, indent=2) + "\n", encoding="utf-8")
    print(output_root)


if __name__ == "__main__":
    main()

