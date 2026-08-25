#!/usr/bin/env python3
"""Validate completed camera originals against a frozen capture manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_FILLED = ("camera_model", "capture_time", "operator_id", "photo_path", "photo_sha256")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.kit_root.resolve()
    manifest = root / "capture_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("capture manifest is empty")

    errors = []
    seen_paths = set()
    for row in rows:
        missing = [field for field in REQUIRED_FILLED if not str(row.get(field, "")).strip()]
        if missing:
            errors.append({"capture_id": row.get("capture_id"), "error": "missing_fields", "fields": missing})
            continue
        photo = Path(row["photo_path"]).expanduser().resolve()
        if not photo.is_file():
            errors.append({"capture_id": row["capture_id"], "error": "missing_photo", "path": str(photo)})
            continue
        if str(photo) in seen_paths:
            errors.append({"capture_id": row["capture_id"], "error": "duplicate_photo_path", "path": str(photo)})
        seen_paths.add(str(photo))
        actual = sha256(photo)
        if actual.lower() != row["photo_sha256"].strip().lower():
            errors.append({"capture_id": row["capture_id"], "error": "photo_hash_mismatch", "actual": actual})

    result = {
        "schema_version": "cta/physical-capture-validation-v1",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "rows": len(rows),
        "unique_photo_paths": len(seen_paths),
        "errors": errors,
        "status": "passed" if not errors else "failed",
    }
    (root / "capture_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
