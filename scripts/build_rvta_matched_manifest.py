#!/usr/bin/env python3
"""Build a seven-condition matched RVTA benchmark from frozen test identifiers."""

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

from cta.rvta_bench import (
    AREA_MATCHED_DIRECT,
    BENIGN_TRUE_EVIDENCE,
    render_area_matched_direct_control,
    render_benign_true_evidence,
)
from cta.strong_attack import BASELINE_POLICY_ID


BASE_ATTACKS = ("none", "naive", "scene_coherent")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    try:
        repo = Path(__file__).resolve().parents[1]
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "not-a-git-checkout"


def with_truth(row: dict, expected_claim: str | None, role: str) -> dict:
    result = {
        key: row[key]
        for key in (
            "sample_id", "source_sha256", "target_label", "attack", "defense", "attack_text",
            "attack_metadata", "image_path",
        )
    }
    result["defense_metadata"] = row.get("defense_metadata", {})
    result["expected_claim"] = expected_claim
    result["condition_role"] = role
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--base-log", type=Path, required=True)
    parser.add_argument("--strong-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--selected-policy", default="v2-telemetry-plaque-compact")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.source_manifest.resolve()
    base_path = args.base_log.resolve()
    strong_path = args.strong_manifest.resolve()
    split_path = args.split_manifest.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    sources = {row["sample_id"]: row for row in json.loads(source_path.read_text(encoding="utf-8"))}
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("active_split") != "test":
        raise ValueError("matched benchmark requires the frozen test split")
    active_ids = list(split["active_ids"])
    if set(split["discovery_ids"]) & set(active_ids):
        raise ValueError("discovery/test leakage")
    if len(active_ids) != len(set(active_ids)):
        raise ValueError("duplicate active test identifiers")

    base_rows = {
        (row["sample_id"], row["attack"]): row
        for row in read_jsonl(base_path)
        if row.get("defense") == "none" and row.get("attack") in BASE_ATTACKS
    }
    strong_rows = {
        (row["sample_id"], row["attack"]): row
        for row in read_jsonl(strong_path)
        if row.get("defense") == "none"
    }
    expected_base = {(sample_id, attack) for sample_id in active_ids for attack in BASE_ATTACKS}
    missing_base = expected_base - set(base_rows)
    if missing_base:
        raise ValueError(f"base log misses {len(missing_base)} frozen-test conditions")
    expected_strong = {
        (sample_id, attack)
        for sample_id in active_ids
        for attack in (BASELINE_POLICY_ID, args.selected_policy)
    }
    missing_strong = expected_strong - set(strong_rows)
    if missing_strong:
        raise ValueError(f"strong manifest misses {len(missing_strong)} frozen-test conditions")

    rows: list[dict] = []
    for sample_id in active_ids:
        source = sources[sample_id]
        source_image = Path(source["image_path"])
        if sha256(source_image) != source["source_sha256"]:
            raise ValueError(f"source hash mismatch: {sample_id}")

        rows.append(with_truth(base_rows[(sample_id, "none")], None, "clean image utility control"))
        rows.append(with_truth(
            base_rows[(sample_id, "naive")], "FALSE", "wrong-object top-banner typography baseline",
        ))
        rows.append(with_truth(
            base_rows[(sample_id, "scene_coherent")], "FALSE", "wrong-object scene-plaque typography baseline",
        ))
        rows.append(with_truth(
            strong_rows[(sample_id, BASELINE_POLICY_ID)], "FALSE", "original causal typography baseline",
        ))
        selected = strong_rows[(sample_id, args.selected_policy)]
        rows.append(with_truth(selected, "FALSE", "frozen evidence-augmented causal typography"))

        reference = selected["attack_metadata"]
        direct = render_area_matched_direct_control(
            source["image_path"], source["target_label"], reference,
            output_root / "images" / AREA_MATCHED_DIRECT / f"{sample_id}.jpg",
        )
        direct_row = {
            "sample_id": sample_id,
            "source_sha256": source["source_sha256"],
            "target_label": source["target_label"],
            "attack": AREA_MATCHED_DIRECT,
            "defense": "none",
            "attack_text": direct.text,
            "attack_metadata": direct.to_dict(),
            "defense_metadata": {},
            "image_path": direct.image_path,
            "expected_claim": direct.expected_claim,
            "condition_role": direct.condition_role,
        }
        rows.append(direct_row)

        benign = render_benign_true_evidence(
            source["image_path"], source["target_label"], reference,
            output_root / "images" / BENIGN_TRUE_EVIDENCE / f"{sample_id}.jpg",
        )
        benign_row = {
            "sample_id": sample_id,
            "source_sha256": source["source_sha256"],
            "target_label": source["target_label"],
            "attack": BENIGN_TRUE_EVIDENCE,
            "defense": "none",
            "attack_text": benign.text,
            "attack_metadata": benign.to_dict(),
            "defense_metadata": {},
            "image_path": benign.image_path,
            "expected_claim": benign.expected_claim,
            "condition_role": benign.condition_role,
        }
        rows.append(benign_row)

    rows.sort(key=lambda row: (row["sample_id"], row["attack"], row["defense"]))
    manifest_path = output_root / "render_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    coverage = {
        "samples": len(active_ids),
        "conditions": sorted({row["attack"] for row in rows}),
        "rows": len(rows),
        "rows_per_sample": len(rows) // len(active_ids),
        "active_ids": active_ids,
        "discovery_ids": split["discovery_ids"],
        "test_ids": split["test_ids"],
    }
    (output_root / "split_manifest.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": "cta/rvta-matched-render-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "host": platform.node(),
        "selected_policy": args.selected_policy,
        "source_manifest": str(source_path),
        "source_manifest_sha256": sha256(source_path),
        "base_log": str(base_path),
        "base_log_sha256": sha256(base_path),
        "strong_manifest": str(strong_path),
        "strong_manifest_sha256": sha256(strong_path),
        "source_split_manifest": str(split_path),
        "source_split_manifest_sha256": sha256(split_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "coverage": coverage,
        "area_control": "exact per-image bbox, placement, palette, and resized canvas from frozen evidence policy",
    }
    (output_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"samples": len(active_ids), "rows": len(rows), "output": str(output_root)}))


if __name__ == "__main__":
    main()
