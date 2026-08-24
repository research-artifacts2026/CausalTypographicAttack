#!/usr/bin/env python3
"""Render registered evidence-augmented CTA candidates on fixed dataset splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.generation import AttackText, AttackTextGenerator
from cta.render import render_attack
from cta.strong_attack import (
    BASELINE_POLICY_ID,
    candidate_policies,
    policy_by_id,
    render_strong_attack,
    split_samples_stratified,
)


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


def append_row(rows: list[dict], sample: dict, attack: str, attack_text: str, image_path: str, metadata: dict) -> None:
    rows.append({
        "sample_id": sample["sample_id"],
        "source_sha256": sample["source_sha256"],
        "target_label": sample["target_label"],
        "attack": attack,
        "defense": "none",
        "attack_text": attack_text,
        "attack_metadata": metadata,
        "defense_metadata": {},
        "image_path": image_path,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("discovery", "test", "ablation", "budgeted_test"), required=True,
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--discovery-samples", type=int, default=12)
    parser.add_argument("--test-samples", type=int, default=100)
    parser.add_argument("--ablation-samples", type=int, default=0)
    parser.add_argument("--budgeted-test-samples", type=int, default=0)
    parser.add_argument("--policy-file", type=Path)
    args = parser.parse_args()

    source_manifest = args.source_manifest.resolve()
    samples = json.loads(source_manifest.read_text(encoding="utf-8"))
    by_id = {row["sample_id"]: row for row in samples}
    splits = split_samples_stratified(
        samples,
        args.seed,
        args.discovery_samples,
        args.test_samples,
        args.ablation_samples,
        args.budgeted_test_samples,
    )
    selected_ids = splits[args.split]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    selection = None
    if args.split in {"discovery", "ablation"}:
        policies = candidate_policies()
    else:
        if not args.policy_file:
            raise ValueError(f"{args.split} rendering requires --policy-file")
        selection = json.loads(args.policy_file.read_text(encoding="utf-8"))
        if args.split == "test":
            if selection.get("split") != "discovery":
                raise ValueError("test policy file is not marked as discovery-only")
            policies = [policy_by_id(selection["selected_policy_id"])]
        else:
            if selection.get("selection_split") != "ablation":
                raise ValueError("budgeted-test policy file must be frozen on the ablation split")
            selected_policy_ids = selection.get("selected_policy_ids", [])
            if not selected_policy_ids:
                raise ValueError("budgeted-test policy file has no selected_policy_ids")
            if len(selected_policy_ids) != len(set(selected_policy_ids)):
                raise ValueError("budgeted-test policy sequence contains duplicates")
            policies = [policy_by_id(policy_id) for policy_id in selected_policy_ids]

    rows: list[dict] = []
    generator = AttackTextGenerator(args.seed)
    for sample_id in selected_ids:
        sample = by_id[sample_id]
        source_path = Path(sample["image_path"])
        if sha256(source_path) != sample["source_sha256"]:
            raise ValueError(f"source image hash mismatch: {sample_id}")
        if args.split == "budgeted_test":
            append_row(
                rows,
                sample,
                "none",
                "",
                str(source_path.resolve()),
                {
                    "policy_id": "none",
                    "condition_role": "clean control",
                    "rendered_sha256": sample["source_sha256"],
                    "overlay_area_fraction": 0.0,
                },
            )
        baseline_text, baseline_family = generator._causal_claim(sample["target_label"])
        baseline = render_attack(
            sample["image_path"],
            AttackText(BASELINE_POLICY_ID, baseline_text, None, baseline_family),
            output_root / "images" / BASELINE_POLICY_ID / f"{sample_id}.jpg",
        )
        with Image.open(baseline.image_path) as baseline_image:
            baseline_pixels = baseline_image.width * baseline_image.height
        baseline_meta = baseline.to_dict()
        baseline_meta.update({
            "policy_id": BASELINE_POLICY_ID,
            "claim_variant": "legacy",
            "artifact_style": "plaque",
            "scale_level": "legacy",
            "rendered_sha256": sha256(Path(baseline.image_path)),
            "overlay_area_fraction": (
                (baseline.bbox[2] - baseline.bbox[0]) * (baseline.bbox[3] - baseline.bbox[1])
                / baseline_pixels
            ) if baseline.bbox else 0.0,
        })
        append_row(rows, sample, BASELINE_POLICY_ID, baseline.text, baseline.image_path, baseline_meta)

        for policy in policies:
            rendered = render_strong_attack(
                sample["image_path"], sample["target_label"], policy,
                output_root / "images" / policy.policy_id / f"{sample_id}.jpg",
            )
            append_row(rows, sample, policy.policy_id, rendered.text, rendered.image_path, rendered.to_dict())

    rows.sort(key=lambda row: (row["sample_id"], row["attack"]))
    manifest_path = output_root / "render_manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    split_record = {
        "seed": args.seed,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "discovery_ids": splits["discovery"],
        "test_ids": splits["test"],
        "ablation_ids": splits["ablation"],
        "budgeted_test_ids": splits["budgeted_test"],
        "active_split": args.split,
        "active_ids": selected_ids,
        "selection": "SHA-256 order within violation family, then deterministic family round-robin",
    }
    (output_root / "split_manifest.json").write_text(json.dumps(split_record, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": "cta/strong-candidate-render-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "host": platform.node(),
        "split": args.split,
        "samples": len(selected_ids),
        "candidate_policies": [policy.to_dict() for policy in policies],
        "budgeted_policy_order": (
            selection.get("selected_policy_ids") if args.split == "budgeted_test" else None
        ),
        "baseline_policy": BASELINE_POLICY_ID,
        "rendered_rows": len(rows),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "policy_file": str(args.policy_file.resolve()) if args.policy_file else None,
        "policy_file_sha256": sha256(args.policy_file.resolve()) if args.policy_file else None,
    }
    (output_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"split": args.split, "samples": len(selected_ids), "rows": len(rows), "output": str(output_root)}))


if __name__ == "__main__":
    main()
