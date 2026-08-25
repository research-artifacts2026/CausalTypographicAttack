#!/usr/bin/env python3
"""Render a predeclared moderate/strong/extreme RVTA severity ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.generation import AttackText
from cta.render import render_attack
from cta.violation_catalog import SEVERITY_LEVELS, claims_for_label


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--severity", action="append", choices=SEVERITY_LEVELS, default=[])
    parser.add_argument("--scenario", action="append", default=[])
    args = parser.parse_args()

    source_manifest = args.source_manifest.resolve()
    source_rows = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("source manifest must be a non-empty JSON list")
    severities = tuple(args.severity or SEVERITY_LEVELS)
    allowed_scenarios = set(args.scenario)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "render_manifest.jsonl"
    if manifest.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {manifest}")

    rows: list[dict] = []
    for source in source_rows:
        sample_id = str(source["sample_id"])
        image_path = Path(source["image_path"]).resolve()
        if sha256(image_path) != source["source_sha256"]:
            raise ValueError(f"source image hash mismatch: {sample_id}")
        rows.append({
            **source,
            "attack": "none",
            "defense": "none",
            "attack_text": "",
            "attack_metadata": {
                "condition_role": "clean control",
                "rendered_sha256": source["source_sha256"],
                "overlay_area_fraction": 0.0,
            },
            "defense_metadata": {},
        })
        for severity in severities:
            for claim in claims_for_label(source["target_label"], severity):
                if allowed_scenarios and claim.scenario_id not in allowed_scenarios:
                    continue
                attack = f"severity-{claim.scenario_id}-{severity}"
                output = output_root / "images" / attack / f"{sample_id}.jpg"
                rendered = render_attack(
                    str(image_path),
                    AttackText(attack, claim.text, None, claim.family),
                    output,
                )
                bbox = rendered.bbox
                area_fraction = None
                if bbox:
                    from PIL import Image
                    with Image.open(rendered.image_path) as image:
                        area_fraction = (
                            (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            / (image.width * image.height)
                        )
                rows.append({
                    **source,
                    "attack": attack,
                    "defense": "none",
                    "attack_text": claim.text,
                    "attack_metadata": {
                        **rendered.to_dict(),
                        **claim.to_dict(),
                        "rendered_sha256": sha256(Path(rendered.image_path)),
                        "overlay_area_fraction": area_fraction,
                    },
                    "defense_metadata": {},
                    "image_path": rendered.image_path,
                })

    rows.sort(key=lambda row: (row["sample_id"], row["attack"]))
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    provenance = {
        "schema_version": "cta/violation-severity-build-v1",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "samples": len(source_rows),
        "severities": list(severities),
        "scenario_filter": sorted(allowed_scenarios),
        "rows": len(rows),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "reporting_boundary": (
            "Ordinary-market-price is a common-sense plausibility anomaly, not a "
            "physical impossibility. All scenarios require independent human validation "
            "of relevance, ambiguity, naturalness, and violation strength before paper use."
        ),
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
