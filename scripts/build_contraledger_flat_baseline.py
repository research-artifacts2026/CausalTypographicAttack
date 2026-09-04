#!/usr/bin/env python3
"""Build a content-matched flat-typography ContraLedger baseline.

The source image, question, symbolic record, registered text, and semantic
answers are copied from a frozen three-state manifest.  Only the delivery
layer changes: true and false records are rendered as a deterministic flat
panel.  No victim output is read by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.contraledger import cue_line, neutral_plan, neutral_record
from cta.contraledger_threeway import CONDITIONS
from cta.question_bench import file_sha256
from cta.scei_attack import CounterfactualRecord, fallback_scene_plan, render_carrier
from scripts.analyze_contraledger_threeway import audit_manifest


_RENDER_FIELDS = {
    "carrier_quad",
    "image_path",
    "image_sha256",
    "mask_path",
    "mask_sha256",
    "overlay_area_fraction",
    "renderer",
}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def strip_render_fields(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in _RENDER_FIELDS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-area-fraction", type=float, default=0.15)
    args = parser.parse_args()

    source_path = args.source_manifest.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable baseline: {output}")
    source_rows = read_jsonl(source_path)
    source_audit = audit_manifest(source_rows)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in source_rows:
        grouped[str(row["item_id"])][str(row["condition"])] = row

    output.mkdir(parents=True)
    built: list[dict] = []
    for item_id in sorted(grouped):
        triplet = grouped[item_id]
        if set(triplet) != set(CONDITIONS):
            raise ValueError(f"{item_id}: incomplete source triplet")
        source = triplet["source_absent"]
        false_source = triplet["record_false"]
        record = neutral_record(CounterfactualRecord(**dict(false_source["record"])))
        plan = neutral_plan(
            fallback_scene_plan(
                str(source["target_label"]), str(source["family"]), item_id
            )
        )
        source_row = {
            **strip_render_fields(source),
            "schema_version": "cta/contraledger-threeway-delivery-item-v1",
            "delivery_method": "source-unmodified",
            "image_path": source["source_path"],
            "image_sha256": source["source_sha256"],
        }
        built.append(source_row)
        for condition, truth in (("record_true", "true"), ("record_false", "false")):
            frozen = triplet[condition]
            image_path = output / "images" / condition / f"{item_id}.jpg"
            mask_path = output / "masks" / condition / f"{item_id}.png"
            # Leave a one-per-mille integer-rasterization margin so the final
            # binary mask remains at or below the registered area cap.
            rendered = render_carrier(
                source["source_path"],
                plan,
                record,
                truth,
                "flat",
                image_path,
                item_id,
                mask_output=mask_path,
                max_area_fraction=args.max_area_fraction - 0.001,
                status_line=cue_line("values_only", item_id),
            ).to_dict()
            if float(rendered["overlay_area_fraction"]) > args.max_area_fraction + 1e-9:
                raise ValueError(f"{item_id}: flat carrier exceeds registered area cap")
            built.append({
                **strip_render_fields(frozen),
                "schema_version": "cta/contraledger-threeway-delivery-item-v1",
                "delivery_method": "naive-flat-matched",
                **rendered,
            })

    order = {condition: index for index, condition in enumerate(CONDITIONS)}
    built.sort(key=lambda row: (str(row["item_id"]), order[str(row["condition"])]))
    manifest = output / "manifest.jsonl"
    write_jsonl(manifest, built)
    output_audit = audit_manifest(built)
    provenance = {
        "schema_version": "cta/contraledger-threeway-delivery-build-v1",
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "delivery_method": "naive-flat-matched",
        "items": len(grouped),
        "rows": len(built),
        "conditions": list(CONDITIONS),
        "family_counts": dict(sorted(Counter(
            triplet["record_false"]["family"] for triplet in grouped.values()
        ).items())),
        "source_manifest": str(source_path),
        "source_manifest_sha256": file_sha256(source_path),
        "source_manifest_audit": source_audit,
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "manifest_audit": output_audit,
        "matched_fields": [
            "source image", "question", "option map", "symbolic record",
            "registered read text", "semantic answers", "family", "item selection",
        ],
        "changed_factor": "delivery renderer only",
        "max_area_fraction": args.max_area_fraction,
        "integer_rasterization_margin": 0.001,
        "selection_policy": "all rows from the frozen source manifest; no outcome filtering",
        "victim_outputs_used": False,
        "claim_boundary": (
            "This is a content-matched flat-renderer baseline, not the original "
            "target-token content used by typographic-attack papers."
        ),
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
