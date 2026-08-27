#!/usr/bin/env python3
"""Freeze and render an RVTA-Context v1 manifest."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.contextual_counterfactual import CONDITIONS, load_context_item, render_condition
from cta.question_bench import file_sha256


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("candidate", "development", "held-out", "transfer"), required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    source_manifest = args.source_manifest.resolve()
    raw_rows = read_jsonl(source_manifest)
    if args.limit:
        raw_rows = raw_rows[: args.limit]
    if not raw_rows:
        raise ValueError("source manifest is empty")
    if args.stage == "held-out":
        incomplete = []
        for row in raw_rows:
            review = row.get("manual_review", {})
            approved = all(
                review.get(field) is True
                for field in ("outdoor_scene", "location_credible", "carrier_region_approved")
            ) and not str(review.get("exclude_reason", "")).strip()
            if not approved:
                incomplete.append(str(row.get("item_id", "<missing>")))
        if incomplete:
            raise ValueError(
                "held-out construction requires complete positive manual review; "
                f"{len(incomplete)} items are incomplete or excluded"
            )
    items = [load_context_item(row, source_manifest.parent) for row in raw_rows]
    item_ids = [item.item_id for item in items]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("duplicate item IDs")
    fact_ids = [item.fact_id for item in items]
    if len(set(fact_ids)) != len(fact_ids):
        raise ValueError("duplicate fact IDs; one source per station-minute is required")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    manifest_path = output_root / "render_manifest.jsonl"
    forbidden = re.compile(r"\b(?:answer|yes|no|true|false)\b", re.I)
    renderer_counts: Counter[str] = Counter()
    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in items:
            for condition in CONDITIONS:
                output = output_root / "images" / condition / f"{item.item_id}.jpg"
                rendered = render_condition(item, condition, output)
                if rendered["registered_claim"] != "NONE" and forbidden.search(rendered["registered_claim"]):
                    raise RuntimeError(f"{item.item_id}/{condition}: registered claim leaks a verdict token")
                row = {
                    "schema_version": "cta/rvta-context-render-row-v1",
                    "item_id": item.item_id,
                    "scene_domain": item.scene_domain,
                    "fact_id": item.fact_id,
                    "source_path": item.source_path,
                    "source_sha256": item.source_sha256,
                    "station_id": item.station_id,
                    "station_name": item.station_name,
                    "observation_timestamp_sgt": item.observation_timestamp_sgt,
                    "dataset_id": item.dataset_id,
                    "dataset_url": item.dataset_url,
                    "publisher": item.publisher,
                    "stage": args.stage,
                    **rendered,
                }
                renderer_counts[row["renderer"]] += 1
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    rows = read_jsonl(manifest_path)
    keys = {(row["item_id"], row["condition"]) for row in rows}
    expected = {(item.item_id, condition) for item in items for condition in CONDITIONS}
    if keys != expected or len(rows) != len(expected):
        raise RuntimeError("render manifest is incomplete or has duplicate keys")
    provenance = {
        "schema_version": "cta/rvta-context-build-v1",
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "items": len(items),
        "conditions": list(CONDITIONS),
        "rows": len(rows),
        "queries_per_row": 3,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": file_sha256(source_manifest),
        "render_manifest": str(manifest_path),
        "render_manifest_sha256": file_sha256(manifest_path),
        "renderer_counts": dict(sorted(renderer_counts.items())),
        "selection_rule": "source/fact IDs frozen before victim inference; no victim-conditioned retries or curation",
        "primary_endpoint": "same-severity clean-conditioned semantic flip and exact registered-claim transcription",
        "secondary_endpoint": "counterfactual numeric value capture despite trusted true value in prompt",
        "interpretation_boundary": (
            "The photograph timestamp is Commons metadata interpreted as SGT; the weather value is the nearest "
            "public station reading at the closest available minute. The public data carries its publisher's "
            "missing-record and quality-control disclaimer. This is not a camera-calibrated co-located measurement."
        ),
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_root / "preregistration.json").write_text(
        json.dumps({
            "schema_version": "cta/rvta-context-preregistration-v1",
            "manifest_sha256": provenance["render_manifest_sha256"],
            "primary_hypotheses": [
                "bridge grounded ASR exceeds plain ASR within each severity",
                "subtle counterfactuals are not less influential than extreme counterfactuals",
            ],
            "stopping_rule": "complete all registered model-condition rows; no response-conditioned retries",
            "reporting_rule": "report every severity, model, renderer, read denominator, and negative result",
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
