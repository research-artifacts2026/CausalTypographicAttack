#!/usr/bin/env python3
"""Convert an existing RVTA-Context collection to portable relative paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.collection_root.resolve()
    manifest = root / "sources.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        source = Path(row["source"]["path"])
        if not source.is_absolute():
            source = root / source
        source = source.resolve()
        if not source.is_file() or root not in source.parents:
            raise ValueError(f"{row['item_id']}: source is missing or outside collection root")
        if sha256(source) != row["source"]["sha256"]:
            raise ValueError(f"{row['item_id']}: source hash mismatch")
        row["source"]["path"] = source.relative_to(root).as_posix()
    temporary = manifest.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(manifest)
    provenance_path = root / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["source_manifest_sha256"] = sha256(manifest)
    provenance["path_policy"] = "source paths are relative to the collection root"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"items": len(rows), "manifest_sha256": provenance["source_manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
