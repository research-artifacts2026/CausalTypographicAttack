#!/usr/bin/env python3
"""Create a validated resumable prefix from disjoint partial prediction logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import file_sha256
from scripts.analyze_contraledger import _FROZEN_INPUT_FIELDS


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-log", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prefix: {output}")
    manifest_rows = read_jsonl(manifest_path)
    expected = {(str(row["item_id"]), str(row["condition"])): row for row in manifest_rows}
    if len(expected) != len(manifest_rows):
        raise ValueError("manifest has duplicate keys")

    keyed: dict[tuple[str, str], dict] = {}
    inputs = []
    for raw_path in args.input_log:
        path = raw_path.resolve()
        rows = read_jsonl(path)
        for row in rows:
            key = (str(row["item_id"]), str(row["condition"]))
            if key in keyed:
                raise ValueError(f"duplicate partial key: {key}")
            if key not in expected:
                raise ValueError(f"unregistered partial key: {key}")
            frozen = expected[key]
            for field in _FROZEN_INPUT_FIELDS:
                if row.get(field) != frozen.get(field):
                    raise ValueError(f"{key}: frozen field changed: {field}")
            keyed[key] = row
        inputs.append({"path": str(path), "rows": len(rows), "sha256": file_sha256(path)})

    condition_order = {
        "values_only_true": 0,
        "values_only_false": 1,
        "authority_true": 2,
        "authority_false": 3,
        "explicit_conclusion_true": 4,
        "explicit_conclusion_false": 5,
    }
    ordered = [
        keyed[key]
        for key in sorted(keyed, key=lambda key: (key[0], condition_order[key[1]]))
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered),
        encoding="utf-8",
    )
    report = {
        "schema_version": "cta/contraledger-rebased-prefix-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "inputs": inputs,
        "rows": len(ordered),
        "output": str(output),
        "output_sha256": file_sha256(output),
        "policy": "exact disjoint union; duplicate, extra, and frozen-field mismatches are fatal",
    }
    output.with_suffix(".provenance.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
