#!/usr/bin/env python3
"""Build one camera-degradation profile while preserving paired conditions."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import file_sha256
from cta.simulated_capture import PROFILES, simulate_capture


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    source_manifest = args.manifest.resolve()
    rows = read_jsonl(source_manifest)
    if not rows:
        raise ValueError("source manifest is empty")
    expected = {(str(row["question_id"]), str(row["condition"])) for row in rows}
    if len(expected) != len(rows):
        raise ValueError("source manifest has duplicate question-condition rows")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_manifest = output_root / "render_manifest.jsonl"
    if output_manifest.exists():
        raise FileExistsError(f"refusing to overwrite {output_manifest}")
    profile = PROFILES[args.profile]
    with output_manifest.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: (str(item["question_id"]), str(item["condition"]))):
            qid, condition = str(row["question_id"]), str(row["condition"])
            digest_seed = int.from_bytes(
                __import__("hashlib").sha256(f"{args.seed}:{qid}".encode()).digest()[:8], "big",
            )
            output = output_root / "images" / condition / f"{qid}.jpg"
            metadata = simulate_capture(row["image_path"], output, profile, digest_seed)
            result = {
                **row,
                "base_image_path": row["image_path"],
                "base_image_sha256": row["image_sha256"],
                "image_path": str(output),
                "image_sha256": file_sha256(output),
                "capture_profile": args.profile,
                "capture_metadata": metadata,
            }
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    provenance = {
        "schema_version": "cta/simulated-capture-build-v1",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reporting_label": "simulated camera degradation (not a physical capture)",
        "profile": profile.__dict__,
        "seed": args.seed,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": file_sha256(source_manifest),
        "rows": len(rows),
        "output_manifest_sha256": file_sha256(output_manifest),
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()

