#!/usr/bin/env python3
"""Render CTA claims with SceneTAP's public TextDiffuser component.

This is deliberately a renderer-only comparison.  It reuses exact claims and
fixed regions from the controlled PIL run; it does not claim to reproduce the
full SceneTAP planner, which requires a separate multimodal API endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "not-a-git-checkout"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-log", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenetap-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--candidate-index", type=int, default=0)
    args = parser.parse_args()

    source_path = args.source_log.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"render_manifest_shard{args.shard_index:02d}.jsonl"

    rows = read_jsonl(source_path)
    clean_by_id = {
        row["sample_id"]: row for row in rows if row["attack"] == "none" and row["defense"] == "none"
    }
    causal = sorted(
        (row for row in rows if row["attack"] == "causal" and row["defense"] == "none"),
        key=lambda row: row["sample_id"],
    )[: args.limit]
    selected = [row for index, row in enumerate(causal) if index % args.num_shards == args.shard_index]
    completed = {row["sample_id"] for row in read_jsonl(output_path)} if output_path.exists() else set()

    os.chdir(args.scenetap_root)
    sys.path.insert(0, str(args.scenetap_root))
    from utils.text_diffuser import TextDiffuser

    renderer = TextDiffuser()
    provenance = {
        "schema_version": "cta/scenetap-textdiffuser-render-v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "source_log": str(source_path),
        "source_log_sha256": sha256_file(source_path),
        "renderer": "SceneTAP TextDiffuser component",
        "full_scenetap_planner_used": False,
        "selection_rule": f"fixed candidate index {args.candidate_index}; no manual selection",
        "scenetap_git_head": git_head(args.scenetap_root),
        "limit_before_sharding": args.limit,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
    }
    (output_root / f"provenance_shard{args.shard_index:02d}.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8",
    )

    for source in selected:
        if source["sample_id"] in completed:
            continue
        clean = clean_by_id[source["sample_id"]]
        bbox = source["attack_metadata"]["bbox"]
        if not bbox or len(bbox) != 4:
            raise ValueError(f"missing render bbox for {source['sample_id']}")
        points = [(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))]
        claim = source["attack_text"]
        prompt = (
            f"A natural photograph containing a {source['target_label']} and a realistic small sign "
            f"that reads exactly: {claim}"
        )
        started = time.time()
        result = renderer.generate(
            points, clean["image_path"], claim, prompt,
            radio="Two Points", scale_factor=2, regional_diffusion=True,
        )
        candidates = result[0]
        if not candidates or args.candidate_index >= len(candidates):
            raise RuntimeError(f"renderer returned insufficient candidates for {source['sample_id']}")
        image_path = output_root / "images" / f"{source['sample_id']}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        candidates[args.candidate_index].convert("RGB").save(image_path, quality=95)
        record = {
            "sample_id": source["sample_id"],
            "source_sha256": source["source_sha256"],
            "target_label": source["target_label"],
            "attack": "causal_textdiffuser",
            "defense": "none",
            "attack_text": claim,
            "attack_metadata": {
                **source["attack_metadata"],
                "attack": "causal_textdiffuser",
                "renderer": "SceneTAP TextDiffuser component",
                "bbox": bbox,
                "candidate_index": args.candidate_index,
                "candidate_count": len(candidates),
                "prompt": prompt,
            },
            "defense_metadata": {},
            "image_path": str(image_path),
            "render_sha256": sha256_file(image_path),
            "render_latency_s": round(time.time() - started, 4),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        append_jsonl(output_path, record)
        completed.add(source["sample_id"])

    provenance["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["completed_images"] = len(completed)
    (output_root / f"provenance_shard{args.shard_index:02d}.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
