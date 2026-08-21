#!/usr/bin/env python3
"""Freeze one OCR-resilient SceneTAP carrier using discovery results only."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-summary", type=Path, required=True)
    parser.add_argument("--render-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.discovery_summary.resolve()
    provenance_path = args.render_provenance.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen policy: {output_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("split") != "discovery":
        raise ValueError("render provenance is not discovery-only")
    rows = summary.get("styles", [])
    if not rows or any(row.get("strict_asr") is None for row in rows):
        raise ValueError("discovery summary has no eligible style results")
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["strict_asr"]),
            -float(row["complete_transcription_rate"]),
            -float(row["mean_carrier_survival_fraction"]),
            float(row["mean_detector_token_recall"]),
            row["style_id"],
        ),
    )
    selected = ranked[0]
    record = {
        "schema_version": "cta/ocr-resilient-policy-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_split": "discovery",
        "selected_style_id": selected["style_id"],
        "selection_rule": "maximize strict post-RapidOCR ASR; tie-break exact transcription, carrier survival, lower detector recall, style id",
        "selected_discovery_metrics": selected,
        "all_discovery_metrics": ranked,
        "discovery_summary": str(summary_path),
        "discovery_summary_sha256": sha256(summary_path),
        "render_provenance": str(provenance_path),
        "render_provenance_sha256": sha256(provenance_path),
        "test_results_consulted": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
