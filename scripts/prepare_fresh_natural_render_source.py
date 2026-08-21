#!/usr/bin/env python3
"""Convert a globally fresh COCO manifest into paired clean/compact rows.

The compact render supplies only a deterministic claim and typography bbox to
the local SceneTAP TextDiffuser component.  The v4 attack later uses the
original clean image as its global pixel base.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.generation import AttackText, AttackTextGenerator
from cta.render import render_attack


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "not-a-git-checkout"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source_path = args.fresh_manifest.resolve()
    output_root = args.output_root.resolve()
    manifest_path = output_root / "render_manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_path}")
    rows = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("fresh manifest must be a non-empty JSON list")
    if len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise ValueError("fresh manifest contains duplicate sample IDs")

    output_rows = []
    for source in rows:
        image_path = Path(source["image_path"]).resolve()
        if sha256(image_path) != source["source_sha256"]:
            raise ValueError(f"source hash mismatch for {source['sample_id']}")
        clean_row = {
            "sample_id": source["sample_id"],
            "source_sha256": source["source_sha256"],
            "target_label": source["target_label"],
            "attack": "none",
            "defense": "none",
            "attack_text": None,
            "attack_metadata": {"fresh_source_record": source},
            "defense_metadata": {},
            "image_path": str(image_path),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        output_rows.append(clean_row)
        claim, violation = AttackTextGenerator.compact_causal_claim(source["target_label"])
        spec = AttackText("causal_compact", claim, None, violation)
        artifact = render_attack(
            str(image_path),
            spec,
            output_root / "images" / f"{source['sample_id']}.jpg",
        )
        output_rows.append({
            "sample_id": source["sample_id"],
            "source_sha256": source["source_sha256"],
            "target_label": source["target_label"],
            "attack": "causal_compact",
            "defense": "none",
            "attack_text": claim,
            "attack_metadata": {**artifact.to_dict(), "fresh_source_record": source},
            "defense_metadata": {},
            "image_path": artifact.image_path,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    provenance = {
        "schema_version": "cta/fresh-natural-render-source-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "git_head": git_head(),
        "fresh_manifest": str(source_path),
        "fresh_manifest_sha256": sha256(source_path),
        "output_manifest": str(manifest_path),
        "output_manifest_sha256": sha256(manifest_path),
        "samples": len(rows),
        "selection": "all rows from an already frozen globally fresh manifest; no model-output filtering",
        "claim_policy": "deterministic compact claim by violation family",
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"samples": len(rows), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
