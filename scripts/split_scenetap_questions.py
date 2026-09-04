#!/usr/bin/env python3
"""Deterministically shard a frozen SceneTAP question array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    if args.shards <= 0:
        raise ValueError("shards must be positive")
    rows = json.loads(args.questions.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("questions must be a non-empty JSON array")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = []
    for shard in range(args.shards):
        subset = [row for index, row in enumerate(rows) if index % args.shards == shard]
        path = args.output_dir / f"questions_shard_{shard:02d}_of_{args.shards:02d}.json"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        path.write_text(json.dumps(subset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        counts.append(len(subset))
    print(json.dumps({"items": len(rows), "shards": args.shards, "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
