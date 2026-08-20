#!/usr/bin/env python3
"""Create a file-hash-normalized source manifest without mutating prior evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()

    source = args.input.resolve()
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("source manifest must be a non-empty JSON list")
    normalized = []
    mismatches = 0
    for row in rows:
        path = Path(row["image_path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            image.verify()
        actual = sha256(path)
        original = str(row["source_sha256"])
        copy = dict(row)
        copy["source_sha256"] = actual
        copy["upstream_image_bytes_sha256"] = original
        copy["source_hash_kind"] = "materialized-file-bytes"
        normalized.append(copy)
        mismatches += int(actual != original)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
    record = {
        "schema_version": "cta/materialized-source-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(source),
        "input_manifest_sha256": sha256(source),
        "output_manifest": str(args.output.resolve()),
        "output_manifest_sha256": sha256(args.output.resolve()),
        "rows": len(normalized),
        "file_vs_upstream_hash_mismatches": mismatches,
        "policy": "retain the prior hash as upstream_image_bytes_sha256 and use exact materialized file bytes for rendering",
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(normalized), "mismatches": mismatches, "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
