#!/usr/bin/env python3
"""Create matched compact-claim PIL inputs for the natural-render comparison."""

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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "not-a-git-checkout"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-log", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    rows = read_jsonl(args.source_log.resolve())
    clean = {row["sample_id"]: row for row in rows if row["attack"] == "none" and row["defense"] == "none"}
    source_causal = sorted(
        (row for row in rows if row["attack"] == "causal" and row["defense"] == "none"),
        key=lambda row: row["sample_id"],
    )[: args.limit]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for source in source_causal:
        clean_row = clean[source["sample_id"]]
        output_rows.append(clean_row)
        claim, violation = AttackTextGenerator.compact_causal_claim(source["target_label"])
        spec = AttackText("causal_compact", claim, None, violation)
        artifact = render_attack(
            clean_row["image_path"], spec, output_root / "images" / f"{source['sample_id']}.jpg",
        )
        output_rows.append({
            "sample_id": source["sample_id"], "source_sha256": source["source_sha256"],
            "target_label": source["target_label"], "attack": "causal_compact", "defense": "none",
            "attack_text": claim, "attack_metadata": artifact.to_dict(), "defense_metadata": {},
            "image_path": artifact.image_path, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
    manifest = output_root / "render_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    (output_root / "provenance.json").write_text(json.dumps({
        "schema_version": "cta/natural-render-source-v1", "source_log": str(args.source_log.resolve()),
        "source_log_sha256": sha256_file(args.source_log.resolve()), "git_head": git_head(),
        "hostname": platform.node(), "samples": len(source_causal),
        "selection": "lexicographically first sample IDs from completed COCO-300 run",
        "claim_policy": "deterministic compact equivalent by violation family",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"samples": len(source_causal), "manifest": str(manifest)}))


if __name__ == "__main__":
    main()
