#!/usr/bin/env python3
"""Create a method-blinded, matched-sample human evaluation package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path


RATING_COLUMNS = ["legibility_1to5", "visual_integration_1to5", "scene_fit_1to5", "impossibility_1to5"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def opaque_id(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()[:12]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pil-log", type=Path, required=True)
    parser.add_argument("--compact-pil-log", type=Path, required=True)
    parser.add_argument("--textdiffuser-log", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--annotators", type=int, default=3)
    parser.add_argument("--duplicate-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260312)
    args = parser.parse_args()

    pil_rows = read_jsonl(args.pil_log)
    compact_rows = read_jsonl(args.compact_pil_log)
    natural_rows = read_jsonl(args.textdiffuser_log)
    methods: dict[str, dict[str, dict]] = {}
    for method in ("naive", "scene_coherent"):
        methods[method] = {
            row["sample_id"]: row for row in pil_rows
            if row["attack"] == method and row["defense"] == "none"
        }
    methods["causal_compact_pil"] = {
        row["sample_id"]: row for row in compact_rows if row["attack"] == "causal_compact"
    }
    methods["causal_compact_textdiffuser"] = {row["sample_id"]: row for row in natural_rows}
    shared_ids = sorted(set.intersection(*(set(rows) for rows in methods.values())))
    rng = random.Random(args.seed)
    rng.shuffle(shared_ids)
    shared_ids = shared_ids[: args.samples]
    if len(shared_ids) < args.samples:
        raise ValueError(f"requested {args.samples} matched samples, found {len(shared_ids)}")

    output_root = args.output_root.resolve()
    images_root = output_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    private_rows = []
    public_items = []
    for sample_id in shared_ids:
        for method, by_id in methods.items():
            source = by_id[sample_id]
            item_id = opaque_id(args.seed, f"{sample_id}:{method}")
            suffix = Path(source["image_path"]).suffix.lower() or ".jpg"
            destination = images_root / f"{item_id}{suffix}"
            shutil.copy2(source["image_path"], destination)
            public_items.append({
                "item_id": item_id,
                "image": f"images/{destination.name}",
                "claim": source["attack_text"],
            })
            private_rows.append({
                "item_id": item_id, "sample_id": sample_id, "method": method,
                "source_image": source["image_path"], "claim": source["attack_text"],
            })

    fields = ["row_id", "item_id", "image", "claim", *RATING_COLUMNS, "comments"]
    for annotator_index in range(args.annotators):
        order = list(public_items)
        rng_i = random.Random(args.seed + 1000 + annotator_index)
        rng_i.shuffle(order)
        duplicate_count = round(len(order) * args.duplicate_rate)
        duplicates = rng_i.sample(order, duplicate_count)
        assignment = order + [{**row, "duplicate": True} for row in duplicates]
        rng_i.shuffle(assignment)
        rows = []
        for index, item in enumerate(assignment, start=1):
            rows.append({
                "row_id": f"A{annotator_index + 1}-{index:04d}",
                "item_id": item["item_id"], "image": item["image"], "claim": item["claim"],
                **{column: "" for column in RATING_COLUMNS}, "comments": "",
            })
        write_csv(output_root / "assignments" / f"annotator_{annotator_index + 1}.csv", rows, fields)

    write_csv(
        output_root / "private_method_key.csv", private_rows,
        ["item_id", "sample_id", "method", "source_image", "claim"],
    )
    (output_root / "README.md").write_text(
        "# Independent blind evaluation\n\n"
        "Use at least three annotators who did not generate the examples. Keep `private_method_key.csv` hidden until all ratings are locked. "
        "Each annotator completes their CSV independently without discussion. Open the relative image path and score every item: "
        "legibility, visual integration, scene fit, and claim impossibility, each from 1 (very low) to 5 (very high). "
        "Repeated item IDs are intentional attention/reliability checks; do not merge them. Do not alter item IDs. "
        "Place completed files under `responses/` and run `scripts/analyze_human_eval.py`.\n",
        encoding="utf-8",
    )
    (output_root / "provenance.json").write_text(json.dumps({
        "schema_version": "cta/human-eval-pack-v1", "seed": args.seed,
        "matched_samples": len(shared_ids), "methods": sorted(methods),
        "annotators": args.annotators, "duplicate_rate": args.duplicate_rate,
        "ratings": RATING_COLUMNS,
        "status": "awaiting independent human annotations",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"items": len(public_items), "assignments": args.annotators, "output": str(output_root)}))


if __name__ == "__main__":
    main()
