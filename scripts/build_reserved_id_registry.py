#!/usr/bin/env python3
"""Freeze and cross-check the sample IDs consumed by prior model queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SAMPLE_ID = re.compile(r"^coco-\d{12}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "not-a-git-checkout"


def collect_ids(value) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        if isinstance(value.get("sample_id"), (str, int)):
            found.add(str(value["sample_id"]))
        for key, item in value.items():
            if key.endswith("_ids") and isinstance(item, list):
                found.update(str(sample_id) for sample_id in item)
            elif key != "sample_id":
                found.update(collect_ids(item))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                found.update(collect_ids(item))
            elif isinstance(item, (str, int)) and str(item).startswith("coco-"):
                found.add(str(item))
    return {sample_id for sample_id in found if SAMPLE_ID.fullmatch(sample_id)}


def values_from_file(path: Path) -> list:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return [json.loads(path.read_text(encoding="utf-8"))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-canonical-sha256", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite frozen reserved-ID registry")

    manifest_ids: set[str] = set()
    manifest_rows = []
    for source in args.source_manifest:
        path = source.resolve()
        ids: set[str] = set()
        for value in values_from_file(path):
            ids.update(collect_ids(value))
        if not ids:
            raise ValueError(f"source manifest contains no sample IDs: {path}")
        manifest_ids.update(ids)
        manifest_rows.append({"path": str(path), "sha256": sha256(path), "sample_ids": len(ids)})

    runs_root = args.runs_root.resolve()
    query_ids: set[str] = set()
    query_rows = []
    for path in sorted(runs_root.glob("**/*.jsonl")):
        values = values_from_file(path)
        ids: set[str] = set()
        for value in values:
            ids.update(collect_ids(value))
        if ids:
            query_ids.update(ids)
            query_rows.append({"path": str(path.resolve()), "sha256": sha256(path), "sample_ids": len(ids)})

    if not query_ids:
        raise ValueError("no prior target-model query artifacts were discovered")
    missing_from_manifests = sorted(query_ids - manifest_ids)
    unqueried_in_manifests = sorted(manifest_ids - query_ids)
    if missing_from_manifests or unqueried_in_manifests:
        raise ValueError(
            "reserved-ID coverage mismatch: "
            f"query_only={len(missing_from_manifests)}, manifest_only={len(unqueried_in_manifests)}"
        )
    reserved = sorted(manifest_ids)
    canonical_bytes = ("\n".join(reserved) + "\n").encode("utf-8")
    canonical_sha = hashlib.sha256(canonical_bytes).hexdigest()
    if len(reserved) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} reserved IDs, found {len(reserved)}")
    if canonical_sha != args.expected_canonical_sha256:
        raise ValueError(
            f"canonical ID hash mismatch: expected {args.expected_canonical_sha256}, found {canonical_sha}"
        )
    generator_path = Path(__file__).resolve()
    fresh_builder = generator_path.with_name("build_fresh_reality_manifest.py")
    record = {
        "schema_version": "cta/global-coco-reserved-registry-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_git_head": git_head(generator_path.parents[1]),
        "generator_sha256": sha256(generator_path),
        "build_fresh_script_sha256": sha256(fresh_builder),
        "id_format": "^coco-[0-9]{12}$",
        "canonicalization": "UTF-8 lexicographically sorted unique IDs, newline separated, terminal newline",
        "runs_root": str(runs_root),
        "source_manifests": manifest_rows,
        "observed_jsonl_artifacts": query_rows,
        "coverage_audit": {
            "manifest_union_count": len(manifest_ids),
            "observed_jsonl_union_count": len(query_ids),
            "observed_minus_reserved": [],
            "reserved_minus_observed": []
        },
        "canonical_ids_sha256": canonical_sha,
        "reserved_id_count": len(reserved),
        "reserved_ids": reserved
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "reserved_ids": len(reserved),
        "source_manifests": len(manifest_rows),
        "observed_jsonl_artifacts": len(query_rows),
        "canonical_ids_sha256": record["canonical_ids_sha256"]
    }, indent=2))


if __name__ == "__main__":
    main()
