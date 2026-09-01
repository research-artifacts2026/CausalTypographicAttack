#!/usr/bin/env python3
"""Audit and analyze preregistered Causal-Bridge mechanism-control runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.bridge_mechanism_controls import (
    ALL_CONDITIONS,
    SCHEMA_VERSION,
    clean_eligible_ids,
    clustered_binary_interaction_model,
    clustered_bootstrap_mean,
    interaction_contributions,
    parse_semantic_answer,
    read_jsonl,
    summarize_conditions,
    transcription_fields_match,
    validate_manifest_rows,
)
from cta.question_bench import file_sha256


def parse_assignment(value: str) -> tuple[str, str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or "@" not in label:
        raise ValueError("--run must use MODEL@DATASET=PATH")
    model, dataset = label.rsplit("@", 1)
    if not model.strip() or not dataset.strip() or not raw_path.strip():
        raise ValueError("--run must use non-empty MODEL@DATASET=PATH values")
    return model.strip(), dataset.strip(), Path(raw_path).resolve()


def validate_run(model: str, dataset: str, predictions_path: Path) -> tuple[list[dict], dict]:
    rows = read_jsonl(predictions_path)
    provenance_path = predictions_path.parent / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("status") != "complete" or int(provenance.get("completed_rows", -1)) != len(rows):
        raise ValueError(f"{model}@{dataset}: run provenance is incomplete")
    manifest_path = Path(provenance["manifest"]).resolve()
    manifest = read_jsonl(manifest_path)
    manifest_hash = file_sha256(manifest_path)
    if provenance.get("manifest_sha256") != manifest_hash:
        raise ValueError(f"{model}@{dataset}: run provenance manifest hash mismatch")
    validate_manifest_rows(manifest, check_files=True)
    manifest_by_key = {(row["item_id"], row["condition"]): row for row in manifest}
    prediction_keys = [(row["item_id"], row["condition"]) for row in rows]
    if len(prediction_keys) != len(set(prediction_keys)) or set(prediction_keys) != set(manifest_by_key):
        raise ValueError(f"{model}@{dataset}: prediction keys are duplicate or incomplete")
    row_datasets = {str(row["dataset"]).lower() for row in rows}
    if row_datasets != {dataset.lower()}:
        raise ValueError(f"{model}@{dataset}: supplied dataset label differs from log")
    for row in rows:
        key = (row["item_id"], row["condition"])
        source = manifest_by_key[key]
        for field in (
            "image_sha256", "source_sha256", "verification_question", "correct_semantic",
            "target_semantic", "registered_read_fields", "bbox", "placement",
            "overlay_area_fraction",
        ):
            if row.get(field) != source.get(field):
                raise ValueError(f"{model}@{dataset}/{key}: log differs from manifest for {field}")
        if file_sha256(row["image_path"]) != row["image_sha256"]:
            raise ValueError(f"{model}@{dataset}/{key}: current image hash mismatch")
        reparsed = parse_semantic_answer(
            row.get("answer_raw", ""), row["answer_format"], row["option_order"],
        )
        reread = transcription_fields_match(row.get("read_raw", ""), row["registered_read_fields"])
        if row.get("parsed_semantic") != reparsed or bool(row.get("read_match")) != reread:
            raise ValueError(f"{model}@{dataset}/{key}: derived score does not replay")
    build_provenance_path = manifest_path.parent / "build_provenance.json"
    build_provenance = json.loads(build_provenance_path.read_text(encoding="utf-8"))
    if build_provenance.get("manifest_sha256") != manifest_hash or build_provenance.get("status") != "frozen":
        raise ValueError(f"{model}@{dataset}: build provenance is not frozen on this manifest")
    descriptor = {
        "model_label": model,
        "dataset_label": dataset,
        "predictions": str(predictions_path),
        "predictions_sha256": file_sha256(predictions_path),
        "run_provenance": str(provenance_path),
        "run_provenance_sha256": file_sha256(provenance_path),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "build_provenance": str(build_provenance_path),
        "build_provenance_sha256": file_sha256(build_provenance_path),
        "model": provenance.get("model"),
        "model_snapshot": provenance.get("model_snapshot"),
    }
    return rows, descriptor


def pooled_condition_summary(cell_rows: list[tuple[str, list[dict]]]) -> list[dict]:
    output = []
    for condition in ALL_CONDITIONS:
        n_total = 0
        n_eligible = 0
        reads = 0
        successes = 0
        for _, rows in cell_rows:
            eligible = clean_eligible_ids(rows)
            selected = [
                row for row in rows
                if row["condition"] == condition and row["item_id"] in eligible
            ]
            n_total += sum(row["condition"] == condition for row in rows)
            n_eligible += len(selected)
            if condition != "no_attack":
                reads += sum(bool(row.get("read_match")) for row in selected)
                successes += sum(
                    row.get("parsed_semantic") == row.get("target_semantic")
                    and bool(row.get("read_match"))
                    for row in selected
                )
        output.append({
            "condition": condition,
            "n_total": n_total,
            "n_clean_correct": n_eligible,
            "read_rate_clean_conditioned": (
                reads / n_eligible if n_eligible and condition != "no_attack" else None
            ),
            "clean_conditioned_read_gated_target_asr": (
                successes / n_eligible if n_eligible and condition != "no_attack" else None
            ),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="append", required=True, help="Repeat MODEL@DATASET=predictions.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--expected-cells", type=int, default=8)
    args = parser.parse_args()

    assignments = [parse_assignment(value) for value in args.run]
    labels = [f"{model}@{dataset}" for model, dataset, _ in assignments]
    if len(labels) != len(set(labels)):
        raise ValueError("duplicate model-dataset run label")

    cell_rows: list[tuple[str, list[dict]]] = []
    input_descriptors = []
    per_cell = {}
    pooled_contributions = []
    for index, (model, dataset, path) in enumerate(assignments):
        cell = f"{model}@{dataset}"
        rows, descriptor = validate_run(model, dataset, path)
        cell_rows.append((cell, rows))
        input_descriptors.append(descriptor)
        contributions = interaction_contributions(rows, cell=cell, dataset=dataset)
        pooled_contributions.extend(contributions)
        interaction = clustered_bootstrap_mean(
            contributions, "interaction", seed=args.seed + index * 100, draws=args.bootstrap_draws,
        )
        per_cell[cell] = {
            "model": model,
            "dataset": dataset,
            "condition_summary": summarize_conditions(rows),
            "primary_interaction_bootstrap": interaction,
            "aligned_minus_reversed_bootstrap": clustered_bootstrap_mean(
                contributions,
                "aligned_minus_reversed",
                seed=args.seed + index * 100 + 1,
                draws=args.bootstrap_draws,
            ),
            "aligned_minus_target_only_bootstrap": clustered_bootstrap_mean(
                contributions,
                "aligned_minus_target_only",
                seed=args.seed + index * 100 + 2,
                draws=args.bootstrap_draws,
            ),
        }

    pooled_primary = clustered_bootstrap_mean(
        pooled_contributions, "interaction", seed=args.seed + 9000, draws=args.bootstrap_draws,
    )
    pooled_reversed = clustered_bootstrap_mean(
        pooled_contributions,
        "aligned_minus_reversed",
        seed=args.seed + 9001,
        draws=args.bootstrap_draws,
    )
    pooled_target_only = clustered_bootstrap_mean(
        pooled_contributions,
        "aligned_minus_target_only",
        seed=args.seed + 9002,
        draws=args.bootstrap_draws,
    )
    positive_cells = sum(
        result["primary_interaction_bootstrap"]["estimate"] is not None
        and result["primary_interaction_bootstrap"]["estimate"] > 0
        for result in per_cell.values()
    )
    complete_cell_plan = len(per_cell) == args.expected_cells
    lower = pooled_primary["ci95"][0]
    if not complete_cell_plan:
        gate = "incomplete_cell_plan_no_mechanism_claim"
    elif lower is not None and lower > 0 and positive_cells >= 6:
        gate = "mechanism_claim_gate_supported"
    else:
        gate = "mechanism_claim_gate_not_supported"

    evidence = {
        "schema_version": f"{SCHEMA_VERSION}/analysis",
        "analysis_status": "complete" if complete_cell_plan else "partial",
        "preregistered_primary_endpoint": "pooled clean-conditioned read-gated target ASR",
        "preregistered_primary_interaction": (
            "(bridge_aligned - bridge_neutral) - (target_only - neutral_only)"
        ),
        "condition_summary_pooled": pooled_condition_summary(cell_rows),
        "primary_interaction_bootstrap": pooled_primary,
        "primary_interaction_binary_model": clustered_binary_interaction_model(
            pooled_contributions
        ),
        "secondary_paired_contrasts": {
            "bridge_aligned_minus_bridge_reversed": pooled_reversed,
            "bridge_aligned_minus_target_only": pooled_target_only,
        },
        "per_model_dataset_cell": per_cell,
        "claim_gate": {
            "status": gate,
            "expected_model_dataset_cells": args.expected_cells,
            "observed_model_dataset_cells": len(per_cell),
            "positive_interaction_cells": positive_cells,
            "required_positive_cells": 6,
            "requires_pooled_bootstrap_interval_above_zero": True,
            "interpretation_if_not_supported": (
                "Do not claim that the false proposition contributes beyond target-semantic "
                "framing; follow the preregistered rename/removal boundary."
            ),
        },
        "bootstrap": {
            "unit": "source item; all model observations for one dataset:item_id stay together",
            "draws": args.bootstrap_draws,
            "seed": args.seed,
            "interval": "paired percentile 95%",
        },
        "inputs": input_descriptors,
        "reporting_boundary": (
            "Experiment A identifies proposition-by-conclusion interaction only; it does not "
            "establish scene dependence, which is reserved for preregistered Experiment B."
        ),
    }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    evidence_path = output / "bridge_mechanism_evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output / "condition_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "cell", "model", "dataset", "condition", "n_total", "n_clean_correct",
            "read_rate_clean_conditioned", "clean_conditioned_read_gated_target_asr",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell, result in per_cell.items():
            for row in result["condition_summary"]:
                writer.writerow({
                    "cell": cell,
                    "model": result["model"],
                    "dataset": result["dataset"],
                    **row,
                })
    with (output / "interaction_by_cell.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["cell", "model", "dataset", "n", "estimate", "ci95_low", "ci95_high"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell, result in per_cell.items():
            value = result["primary_interaction_bootstrap"]
            writer.writerow({
                "cell": cell,
                "model": result["model"],
                "dataset": result["dataset"],
                "n": value["observations"],
                "estimate": value["estimate"],
                "ci95_low": value["ci95"][0],
                "ci95_high": value["ci95"][1],
            })

    module_path = Path(__file__).resolve().parents[1] / "cta" / "bridge_mechanism_controls.py"
    analysis_provenance = {
        "schema_version": f"{SCHEMA_VERSION}/analysis-provenance",
        "status": "complete" if complete_cell_plan else "partial",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script_sha256": file_sha256(Path(__file__)),
        "mechanism_module_sha256": file_sha256(module_path),
        "evidence_sha256": file_sha256(evidence_path),
        "input_prediction_sha256": {
            f"{row['model_label']}@{row['dataset_label']}": row["predictions_sha256"]
            for row in input_descriptors
        },
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
        "claim_gate_status": gate,
    }
    (output / "analysis_provenance.json").write_text(
        json.dumps(analysis_provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "cells": len(per_cell),
        "claim_gate": gate,
        "evidence": str(evidence_path),
    }, indent=2))


if __name__ == "__main__":
    main()
