#!/usr/bin/env python3
"""Build a frozen, truth/order/format-balanced RVTA-QA manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import file_sha256
from cta.rvta_qa_balanced import (
    CONDITIONS,
    assign_counterbalance,
    build_spec,
    read_prompt_for,
    render_condition,
    split_samples,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-manifest", type=Path, action="append", required=True,
        help="JSON item list; repeat to merge disjoint frozen source registries.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--stage", choices=("development", "held-out", "transfer"), required=True)
    parser.add_argument("--allow-source-reencoding", action="store_true")
    args = parser.parse_args()

    source_manifests = [path.resolve() for path in args.sample_manifest]
    samples = []
    source_descriptors = []
    for source_manifest in source_manifests:
        loaded = json.loads(source_manifest.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError("every sample manifest must be a JSON list")
        samples.extend(loaded)
        source_descriptors.append({
            "path": str(source_manifest),
            "sha256": file_sha256(source_manifest),
            "items": len(loaded),
        })
    source_ids = [str(row.get("sample_id", row.get("item_id", ""))) for row in samples]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("sample manifests contain duplicate item IDs")
    selected = split_samples(samples, args.seed, args.offset, args.limit)
    assigned = assign_counterbalance(selected, args.seed)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    manifest_path = output_root / "render_manifest.jsonl"
    item_rows = []
    with manifest_path.open("w", encoding="utf-8") as handle:
        for sample, cell in assigned:
            spec = build_spec(
                sample,
                args.dataset,
                cell,
                allow_source_reencoding=args.allow_source_reencoding,
            )
            item_rows.append(spec.to_dict())
            for condition in CONDITIONS:
                output = output_root / "images" / condition / f"{spec.item_id}.jpg"
                rendered = render_condition(spec, condition, output)
                row = {
                    **spec.to_dict(),
                    **rendered,
                    "answers": [spec.correct_answer_token],
                    "target_aliases": [spec.target_answer_token, spec.target_semantic],
                    "read_prompt": read_prompt_for(condition),
                    "seed": args.seed,
                    "stage": args.stage,
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_root / "items.json").write_text(
        json.dumps(item_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    cell_counts = Counter(row["counterbalance_cell"] for row in item_rows)
    if max(cell_counts.values()) - min(cell_counts.values()) > 1:
        raise RuntimeError("counterbalance cell counts differ by more than one")
    provenance = {
        "schema_version": "cta/rvta-qa-balanced-build-v1",
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "dataset": args.dataset,
        "seed": args.seed,
        "offset": args.offset,
        "items": len(item_rows),
        "conditions": list(CONDITIONS),
        "rows": len(item_rows) * len(CONDITIONS),
        "source_manifests": source_descriptors,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "counterbalance_cell_counts": dict(sorted(cell_counts.items())),
        "question_policy": "one byte-identical verification question per item across every image condition",
        "counterbalance_policy": "truth polarity, AB option order, and AB-versus-YES/NO format are frozen before inference",
        "metric_policy": "semantic clean-conditioned target ASR; grounded ASR also requires registered attack-claim transcription",
        "tuning_boundary": "development may diagnose registered conditions; held-out and transfer cannot select or modify them",
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
