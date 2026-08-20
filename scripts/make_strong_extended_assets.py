#!/usr/bin/env python3
"""Generate cross-dataset and defense tables from complete CTA-v2 logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.strong_attack import BASELINE_POLICY_ID


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_labeled(specs: list[str]) -> dict[str, Path]:
    result = {}
    for spec in specs:
        label, raw = spec.split("=", 1)
        if label in result:
            raise ValueError(f"duplicate model label: {label}")
        result[label] = Path(raw).resolve()
    return result


def validate_complete(path: Path, expected: set[tuple[str, str, str]]) -> list[dict]:
    rows = read_jsonl(path)
    observed = {(row["sample_id"], row["attack"], row["defense"]) for row in rows}
    if observed != expected:
        raise ValueError(f"incomplete log {path}: missing={len(expected-observed)} extra={len(observed-expected)}")
    provenance_path = path.with_name("provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("completed_rows") != len(expected) or not provenance.get("finished_at_utc"):
        raise ValueError(f"unfinished provenance: {path}")
    return rows


def rate(rows: list[dict], attack: str, key: str) -> float:
    selected = [row for row in rows if row["attack"] == attack]
    if not selected:
        raise ValueError(f"missing attack {attack}")
    return statistics.fmean(bool(row[key]) for row in selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-evidence", type=Path, required=True)
    parser.add_argument("--secondary-manifest", type=Path, required=True)
    parser.add_argument("--secondary-model-log", action="append", default=[])
    parser.add_argument("--defense-conditions", type=Path)
    parser.add_argument("--defense-model-log", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    primary_path = args.primary_evidence.resolve()
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    selected_policy = primary["selected_policy_id"]
    secondary_manifest = args.secondary_manifest.resolve()
    secondary_rows = read_jsonl(secondary_manifest)
    secondary_expected = {(row["sample_id"], row["attack"], row["defense"]) for row in secondary_rows}
    secondary_logs = parse_labeled(args.secondary_model_log)
    if set(secondary_logs) != set(primary["models"]):
        raise ValueError("secondary model labels must exactly match primary evidence")

    secondary_metrics = {}
    sources = []
    for model, path in secondary_logs.items():
        rows = validate_complete(path, secondary_expected)
        secondary_metrics[model] = {
            "baseline_asr": rate(rows, BASELINE_POLICY_ID, "attack_success"),
            "selected_asr": rate(rows, selected_policy, "attack_success"),
            "selected_grounded": rate(rows, selected_policy, "claim_matches_overlay"),
        }
        sources.append({"role": "secondary", "model": model, "path": str(path), "sha256": sha256(path)})

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cross_table = [
        "% AUTO-GENERATED from complete frozen-policy COCO/VOC logs; do not edit",
        "\\begin{tabular}{lrrrr}",
        "Model & COCO old & COCO evidence & VOC old & VOC evidence \\\\",
        "\\hline",
    ]
    for model, primary_result in primary["models"].items():
        old_coco = primary_result["attacks"][BASELINE_POLICY_ID]["strict_asr"]
        new_coco = primary_result["attacks"][selected_policy]["strict_asr"]
        secondary_result = secondary_metrics[model]
        cross_table.append(
            f"{model} & {100*old_coco:.1f} & {100*new_coco:.1f} & "
            f"{100*secondary_result['baseline_asr']:.1f} & {100*secondary_result['selected_asr']:.1f} \\\\"
        )
    cross_table += ["\\end{tabular}", ""]
    (output_dir / "generated_strong_cross_dataset_table.tex").write_text(
        "\n".join(cross_table), encoding="utf-8"
    )

    defense_metrics = {}
    if args.defense_conditions and args.defense_model_log:
        defense_path = args.defense_conditions.resolve()
        condition_rows = [
            row for row in read_jsonl(defense_path)
            if row["attack"] == selected_policy and row["defense"] == "rapidocr_mask"
        ]
        if len(condition_rows) != primary["test_samples"]:
            raise ValueError("defense conditions do not cover the held-out test split")
        defense_expected = {(row["sample_id"], row["attack"], row["defense"]) for row in read_jsonl(defense_path)}
        detected = statistics.fmean(
            bool(row["defense_metadata"]["overlay_detected_at_0.5_recall"]) for row in condition_rows
        )
        masked_area = statistics.fmean(
            float(row["defense_metadata"]["masked_area_upper_bound_fraction"]) for row in condition_rows
        )
        defense_logs = parse_labeled(args.defense_model_log)
        defense_table = [
            "% AUTO-GENERATED from complete raw and RapidOCR logs; do not edit",
            "\\begin{tabular}{lrrrr}",
            "Model & Raw & Consistency & RapidOCR & OCR detected \\\\",
            "\\hline",
        ]
        for model, path in defense_logs.items():
            if model not in primary["models"]:
                raise ValueError(f"defense model absent from primary evidence: {model}")
            rows = validate_complete(path, defense_expected)
            rapid_asr = rate(rows, selected_policy, "attack_success")
            raw_asr = primary["models"][model]["attacks"][selected_policy]["strict_asr"]
            # The lexical consistency wrapper passes every selected claim,
            # leaving pixels and deterministic inference input unchanged.
            defense_metrics[model] = {
                "raw_asr": raw_asr,
                "consistency_asr": raw_asr,
                "rapidocr_asr": rapid_asr,
                "ocr_detection_rate": detected,
                "mean_masked_area_upper_bound_fraction": masked_area,
            }
            defense_table.append(
                f"{model} & {100*raw_asr:.1f} & {100*raw_asr:.1f} & "
                f"{100*rapid_asr:.1f} & {100*detected:.1f} \\\\"
            )
            sources.append({"role": "rapidocr", "model": model, "path": str(path), "sha256": sha256(path)})
        defense_table += ["\\end{tabular}", ""]
        (output_dir / "generated_strong_defense_table.tex").write_text(
            "\n".join(defense_table), encoding="utf-8"
        )
        sources.append({"role": "defense_conditions", "path": str(defense_path), "sha256": sha256(defense_path)})

    record = {
        "schema_version": "cta/strong-extended-evidence-v1",
        "selected_policy_id": selected_policy,
        "primary_evidence": str(primary_path),
        "primary_evidence_sha256": sha256(primary_path),
        "secondary_manifest": str(secondary_manifest),
        "secondary_manifest_sha256": sha256(secondary_manifest),
        "secondary_metrics": secondary_metrics,
        "defense_metrics": defense_metrics,
        "sources": sources,
    }
    (output_dir / "strong_extended_evidence.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"models": list(secondary_metrics), "defense_models": list(defense_metrics)}))


if __name__ == "__main__":
    main()
