#!/usr/bin/env python3
"""Recompute the public SCEI-Images-300 release and reject path leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_scei_image_eval import analyze_model, read_jsonl, sha256


FORBIDDEN_KEYS = {
    "image_path", "mask_path", "source_path", "source_dataset_manifest",
    "inference_metadata",
}
FORBIDDEN_TEXT = ("/disk2/", "/home/", "C:\\Users\\")
ANALYSIS_KEYS = (
    "items", "rows", "n_clean_false_correct", "clean_eligibility", "conditions",
    "paired_scene_minus_flat", "families", "answer_cells",
)


def validate_release(result_dir: Path) -> dict:
    result_dir = result_dir.resolve()
    analysis = json.loads((result_dir / "public_analysis.json").read_text(encoding="utf-8"))
    release = json.loads((result_dir / "release_manifest.json").read_text(encoding="utf-8"))
    if analysis.get("status") != "complete":
        raise ValueError("public analysis is not complete")
    if release.get("manifest_sha256") != analysis.get("manifest_sha256"):
        raise ValueError("release and analysis manifest hashes differ")
    expected_items = int(analysis["expected_items"])
    validated = {}
    for name, metadata in release["files"].items():
        path = result_dir / metadata["file"]
        if not path.is_file() or sha256(path) != metadata["sha256"]:
            raise ValueError(f"{name}: public prediction hash mismatch")
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in FORBIDDEN_TEXT):
            raise ValueError(f"{name}: public predictions contain a private path marker")
        rows = read_jsonl(path)
        if len(rows) != int(metadata["rows"]):
            raise ValueError(f"{name}: public prediction row count mismatch")
        for row in rows:
            leaked = FORBIDDEN_KEYS.intersection(row)
            if leaked:
                raise ValueError(f"{name}: forbidden public keys: {sorted(leaked)}")
        actual = analyze_model(rows, expected_items)
        expected = analysis["models"][name]
        for key in ANALYSIS_KEYS:
            if actual[key] != expected[key]:
                raise ValueError(f"{name}: recomputed {key} differs from public analysis")
        validated[name] = {"rows": len(rows), "sha256": metadata["sha256"]}
    if set(validated) != set(analysis["models"]):
        raise ValueError("release model set differs from public analysis")
    return {
        "status": "complete",
        "manifest_sha256": analysis["manifest_sha256"],
        "models": validated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_release(args.result_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
