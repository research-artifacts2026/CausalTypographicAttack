#!/usr/bin/env python3
"""Assemble deterministic SoM shard outputs into one audited flat directory."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    expected = {str(row["image"]) for row in questions}
    if len(expected) != len(questions):
        raise ValueError("question array contains duplicate image names")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    found: dict[str, tuple[Path, Path]] = {}
    for root in args.shard_dir:
        for image in root.glob("*.jpg"):
            mask = image.with_suffix(".npy")
            if not mask.is_file():
                raise FileNotFoundError(f"missing mask for {image}")
            if image.name in found:
                raise ValueError(f"duplicate SoM image across shards: {image.name}")
            found[image.name] = (image, mask)
    if set(found) != expected:
        raise ValueError(
            f"SoM coverage mismatch: missing={len(expected-set(found))}, extra={len(set(found)-expected)}"
        )
    args.output_dir.mkdir(parents=True)
    hashes = []
    for name in sorted(expected):
        image, mask = found[name]
        image_out = args.output_dir / name
        mask_out = image_out.with_suffix(".npy")
        shutil.copy2(image, image_out); shutil.copy2(mask, mask_out)
        hashes.append({
            "image": name,
            "image_sha256": sha256(image_out),
            "mask_sha256": sha256(mask_out),
        })
    provenance = {
        "schema_version": "cta/scenetap-som-merge-v1",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "items": len(expected),
        "questions": str(args.questions.resolve()),
        "questions_sha256": sha256(args.questions),
        "shard_dirs": [str(path.resolve()) for path in args.shard_dir],
        "files": hashes,
    }
    path = args.output_dir / "provenance.json"
    path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "items": len(expected), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
