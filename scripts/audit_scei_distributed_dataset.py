#!/usr/bin/env python3
"""Verify every rendered file and correct durable counts for distributed SCEI."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    manifest_path = root / "dataset_manifest.json"
    provenance_path = root / "provenance.json"
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    panel_images = panel_masks = triptychs = 0
    for row in rows:
        for truth in ("false", "true"):
            condition = row["conditions"][truth]
            triptych = Path(condition["triptych_path"])
            if not triptych.is_file() or file_sha256(triptych) != condition["triptych_sha256"]:
                raise RuntimeError(f"triptych hash mismatch: {row['item_id']} {truth}")
            triptychs += 1
            for panel in condition["panels"]:
                image = Path(panel["image_path"])
                mask = Path(panel["mask_path"])
                if not image.is_file() or file_sha256(image) != panel["image_sha256"]:
                    raise RuntimeError(f"panel image hash mismatch: {row['item_id']} {truth}")
                if not mask.is_file() or file_sha256(mask) != panel["mask_sha256"]:
                    raise RuntimeError(f"panel mask hash mismatch: {row['item_id']} {truth}")
                panel_images += 1
                panel_masks += 1
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance.pop("native_panel_images", None)
    provenance.update({
        "native_panel_slots": len(rows) * 3,
        "rendered_panel_images_false_and_corrected": panel_images,
        "rendered_panel_masks_false_and_corrected": panel_masks,
        "triptych_images_false_and_corrected": triptychs,
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "audit_status": "complete",
        "last_audited_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "complete",
        "triplets": len(rows),
        "native_panel_slots": len(rows) * 3,
        "rendered_panel_images_false_and_corrected": panel_images,
        "triptych_images_false_and_corrected": triptychs,
        "dataset_manifest_sha256": file_sha256(manifest_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
