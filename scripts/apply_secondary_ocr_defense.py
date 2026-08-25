#!/usr/bin/env python3
"""Apply an independently implemented OCR mask to frozen attack images.

This script deliberately performs no victim-model queries and no carrier
selection.  It reads the immutable RapidOCR-aware test conditions, applies a
second OCR engine to each *raw* attack image, and writes a new condition file
that can be evaluated with ``run_ocr_resilient_eval.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.ocr_engines import normalize_easyocr_detections
from cta.ocr_resilient import apply_detected_box_mask, token_recall


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-log", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--engine", choices=("easyocr",), default="easyocr")
    parser.add_argument("--languages", nargs="+", default=["en"])
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-margin-px", type=int, default=2)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.score_threshold <= 1:
        raise ValueError("--score-threshold must be in [0, 1]")
    if args.mask_margin_px < 0:
        raise ValueError("--mask-margin-px must be non-negative")

    source_path = args.source_log.resolve()
    output_root = args.output_root.resolve()
    conditions_path = output_root / "conditions.jsonl"
    detections_path = output_root / "detections.jsonl"
    provenance_path = output_root / "provenance.json"
    if conditions_path.exists() or detections_path.exists() or provenance_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable output under {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    import easyocr

    reader = easyocr.Reader(args.languages, gpu=args.gpu, verbose=False)
    source_rows = sorted(read_jsonl(source_path), key=lambda row: (row["sample_id"], row["style_id"]))
    if not source_rows:
        raise ValueError("source log is empty")

    for source in source_rows:
        raw_path = Path(source["raw_image_path"]).resolve()
        carrier_mask = Path(source["attack_metadata"]["carrier_mask_path"]).resolve()
        layout_bbox = source["attack_metadata"]["layout_bbox"]
        raw_results = reader.readtext(str(raw_path), detail=1, paragraph=False)
        detections = normalize_easyocr_detections(raw_results, args.score_threshold)
        defended_path = output_root / "defended" / source["style_id"] / f"{source['sample_id']}.png"
        defense = apply_detected_box_mask(
            str(raw_path), str(carrier_mask), detections, defended_path,
            margin=args.mask_margin_px, clip_bbox=layout_bbox,
        )
        recognized = " ".join(item["text"] for item in detections)
        metadata = {
            **defense,
            "engine": "EasyOCR",
            "engine_version": importlib.metadata.version("easyocr"),
            "languages": list(args.languages),
            "gpu": bool(args.gpu),
            "score_threshold": args.score_threshold,
            "mask_margin_px": args.mask_margin_px,
            "recognized_text": recognized,
            "overlay_token_recall": token_recall(source["attack_text"], recognized),
            "readability_gate_passed": bool(
                source["attack_metadata"].get("readability_gate_passed", True)
            ),
            "source_detector": source.get("defense_metadata", {}).get("engine"),
            "no_victim_queries_used": True,
        }
        condition = {
            **source,
            "schema_version": "cta/secondary-ocr-condition-v1",
            "defense": "easyocr_mask",
            "defense_metadata": metadata,
            "image_path": str(defended_path),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        append_jsonl(conditions_path, condition)
        append_jsonl(detections_path, {
            "sample_id": source["sample_id"],
            "style_id": source["style_id"],
            "attack_text": source["attack_text"],
            **metadata,
        })

    provenance = {
        "schema_version": "cta/secondary-ocr-mask-v1",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_log": str(source_path),
        "source_log_sha256": sha256(source_path),
        "output_conditions": str(conditions_path),
        "output_conditions_sha256": sha256(conditions_path),
        "output_detections_sha256": sha256(detections_path),
        "rows": len(source_rows),
        "engine": "EasyOCR",
        "engine_version": importlib.metadata.version("easyocr"),
        "languages": list(args.languages),
        "gpu": bool(args.gpu),
        "score_threshold": args.score_threshold,
        "mask_margin_px": args.mask_margin_px,
        "selection_boundary": "The second OCR engine is applied after the RapidOCR-only carrier policy is frozen; no victim outputs or EasyOCR outputs select the carrier.",
        "git_head": git_head(),
        "python": platform.python_version(),
        "hostname": platform.node(),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
