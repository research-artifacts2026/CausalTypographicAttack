#!/usr/bin/env python3
"""Validate complete RVTA-QA logs and emit evidence-backed paper assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import file_sha256
from cta.rvta_qa import CONDITIONS, summarize


DISPLAY_MODELS = {
    "Qwen2.5-VL-3B": "Qwen-3B",
    "Qwen2.5-VL-7B": "Qwen-7B",
    "LLaVA-OneVision-1.5-8B": "LLaVA",
    "InternVL2-8B": "InternVL",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def exact_binomial_two_sided(smaller: int, total: int) -> float:
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(smaller + 1)) / (2 ** total)
    return min(1.0, 2 * tail)


def _success(row: dict, grounded: bool) -> bool:
    return row.get("parsed_answer") == "b" and (not grounded or bool(row.get("read_match")))


def paired_test(rows: list[dict], left: str, right: str, grounded: bool = False) -> dict:
    by_condition = {
        condition: {row["item_id"]: row for row in rows if row["condition"] == condition}
        for condition in ("no_attack", left, right)
    }
    eligible = {
        item_id for item_id, row in by_condition["no_attack"].items()
        if row.get("parsed_answer") == "a"
    }
    left_only = right_only = 0
    for item_id in eligible:
        left_success = _success(by_condition[left][item_id], grounded)
        right_success = _success(by_condition[right][item_id], grounded)
        left_only += int(left_success and not right_success)
        right_only += int(right_success and not left_success)
    discordant = left_only + right_only
    return {
        "left": left, "right": right, "metric": (
            "grounded_clean_conditioned_target_asr" if grounded else "clean_conditioned_target_asr"
        ), "n_clean_correct": len(eligible),
        "left_only_successes": left_only, "right_only_successes": right_only,
        "exact_mcnemar_p": exact_binomial_two_sided(min(left_only, right_only), discordant),
    }


def bootstrap_difference(
    rows: list[dict], left: str, right: str, draws: int, seed: int,
    grounded: bool = False,
) -> dict:
    by_condition = {
        condition: {row["item_id"]: row for row in rows if row["condition"] == condition}
        for condition in ("no_attack", left, right)
    }
    ids = sorted(
        item_id for item_id, row in by_condition["no_attack"].items()
        if row.get("parsed_answer") == "a"
    )
    values = [
        float(_success(by_condition[right][item_id], grounded))
        - float(_success(by_condition[left][item_id], grounded))
        for item_id in ids
    ]
    if not values:
        return {"mean": None, "ci95": [None, None], "draws": draws, "seed": seed}
    rng = random.Random(seed)
    estimates = [statistics.fmean(rng.choice(values) for _ in values) for _ in range(draws)]
    estimates.sort()
    low = estimates[round(0.025 * (draws - 1))]
    high = estimates[round(0.975 * (draws - 1))]
    return {"mean": statistics.fmean(values), "ci95": [low, high], "draws": draws, "seed": seed}


def pct(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.1f}"


def family_summary(rows: list[dict]) -> list[dict]:
    """Compute clean-conditioned rates within preregistered claim families."""
    families = sorted({row["family"] for row in rows})
    result = []
    for family in families:
        family_rows = [row for row in rows if row["family"] == family]
        clean = {
            row["item_id"]: row for row in family_rows
            if row["condition"] == "no_attack"
        }
        eligible = {
            item_id for item_id, row in clean.items()
            if row.get("parsed_answer") == "a"
        }
        entry = {"family": family, "n_items": len(clean), "n_clean_correct": len(eligible)}
        for condition in ("benign_true", "direct_answer", "causal_claim", "evidence_cta", "causal_bridge"):
            condition_rows = [
                row for row in family_rows
                if row["condition"] == condition and row["item_id"] in eligible
            ]
            entry[condition] = {
                "target_asr": (
                    sum(row.get("parsed_answer") == "b" for row in condition_rows) / len(condition_rows)
                    if condition_rows else None
                ),
                "grounded_target_asr": (
                    sum(
                        row.get("parsed_answer") == "b" and bool(row.get("read_match"))
                        for row in condition_rows
                    ) / len(condition_rows)
                    if condition_rows else None
                ),
                "read_accuracy": (
                    sum(bool(row.get("read_match")) for row in condition_rows) / len(condition_rows)
                    if condition_rows else None
                ),
            }
        result.append(entry)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-log", action="append", required=True, help="MODEL=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = read_jsonl(manifest_path)
    expected = {(row["item_id"], row["condition"]) for row in manifest}
    if len(expected) != len(manifest):
        raise ValueError("manifest contains duplicate item-condition keys")
    if {row["condition"] for row in manifest} != set(CONDITIONS):
        raise ValueError("manifest condition set differs from the registered protocol")
    image_hashes = {(row["item_id"], row["condition"]): row["image_sha256"] for row in manifest}
    attack_areas = sorted(
        float(row["overlay_area_fraction"])
        for row in manifest
        if row["condition"] != "no_attack"
    )
    evidence = {
        "schema_version": "cta/rvta-qa-analysis-v1",
        "manifest": str(manifest_path), "manifest_sha256": file_sha256(manifest_path),
        "items": len({row["item_id"] for row in manifest}), "manifest_rows": len(manifest),
        "conditions": list(CONDITIONS),
        "metric": "grounded clean-conditioned target ASR",
        "models": {}, "bootstrap_draws": args.bootstrap_draws, "seed": args.seed,
        "render_audit": {
            "attacked_rows": len(attack_areas),
            "mean_overlay_area_fraction": statistics.fmean(attack_areas),
            "median_overlay_area_fraction": statistics.median(attack_areas),
            "minimum_overlay_area_fraction": min(attack_areas),
            "maximum_overlay_area_fraction": max(attack_areas),
        },
    }
    table_rows = []
    for assignment in args.model_log:
        if "=" not in assignment:
            raise ValueError("--model-log must use MODEL=PATH")
        model_name, value = assignment.split("=", 1)
        log_path = Path(value).resolve()
        rows = read_jsonl(log_path)
        keys = {(row["item_id"], row["condition"]) for row in rows}
        if keys != expected or len(rows) != len(expected):
            raise ValueError(f"{model_name}: incomplete or duplicate log")
        for row in rows:
            key = (row["item_id"], row["condition"])
            if row["image_sha256"] != image_hashes[key] or file_sha256(row["image_path"]) != image_hashes[key]:
                raise ValueError(f"{model_name}: image hash mismatch for {key}")
        provenance_path = log_path.parent / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("status") != "complete" or provenance.get("completed_rows") != len(expected):
            raise ValueError(f"{model_name}: incomplete provenance")
        values = summarize(rows)
        by_condition = {row["condition"]: row for row in values}
        tests = [
            paired_test(rows, "causal_claim", "causal_bridge", grounded=True),
            paired_test(rows, "evidence_cta", "causal_bridge", grounded=True),
            paired_test(rows, "direct_answer", "causal_bridge"),
            paired_test(rows, "benign_true", "causal_bridge"),
        ]
        differences = [
            {"left": left, "right": "causal_bridge", "metric": (
                "grounded_clean_conditioned_target_asr"
                if left in {"causal_claim", "evidence_cta"} else
                "clean_conditioned_target_asr"
            ), **bootstrap_difference(
                rows, left, "causal_bridge", args.bootstrap_draws,
                args.seed + int(hashlib.sha256(f"{model_name}:{left}".encode()).hexdigest()[:8], 16),
                grounded=left in {"causal_claim", "evidence_cta"},
            )}
            for left in ("causal_claim", "evidence_cta", "direct_answer", "benign_true")
        ]
        evidence["models"][model_name] = {
            "log": str(log_path), "log_sha256": file_sha256(log_path),
            "provenance": str(provenance_path), "provenance_sha256": file_sha256(provenance_path),
            "summary": values, "family_summary": family_summary(rows),
            "paired_tests": tests, "paired_differences": differences,
        }
        table_rows.append({"model": model_name, **by_condition})

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "rvta_qa_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8",
    )
    lines = [
        "\\begin{tabular}{lrrrrrrr}", "\\toprule",
        "Model & Clean & Benign & Direct & Claim-G & Evidence-G & Bridge-G & Bridge read \\\\",
        "\\midrule",
    ]
    for row in table_rows:
        lines.append(
            f"{DISPLAY_MODELS.get(row['model'], row['model'])} & {pct(row['no_attack']['answer_accuracy'])} & "
            f"{pct(row['benign_true']['clean_conditioned_target_asr'])} & "
            f"{pct(row['direct_answer']['clean_conditioned_target_asr'])} & "
            f"{pct(row['causal_claim']['grounded_clean_conditioned_asr'])} & "
            f"{pct(row['evidence_cta']['grounded_clean_conditioned_asr'])} & "
            f"{pct(row['causal_bridge']['grounded_clean_conditioned_asr'])} & "
            f"{pct(row['causal_bridge']['read_accuracy'])} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    (output / "generated_rvta_qa_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    family_lines = [
        "\\begin{tabular}{llrrrrr}", "\\toprule",
        "Model & Family & $n_c$ & Direct & Claim & Evidence & Bridge \\\\",
        "\\midrule",
    ]
    for model_name, model in evidence["models"].items():
        for family in model["family_summary"]:
            family_lines.append(
                f"{DISPLAY_MODELS.get(model_name, model_name)} & {family['family']} & {family['n_clean_correct']} & "
                f"{pct(family['direct_answer']['target_asr'])} & "
                f"{pct(family['causal_claim']['grounded_target_asr'])} & "
                f"{pct(family['evidence_cta']['grounded_target_asr'])} & "
                f"{pct(family['causal_bridge']['grounded_target_asr'])} \\\\"
            )
    family_lines.extend(("\\bottomrule", "\\end{tabular}"))
    (output / "generated_rvta_qa_family_table.tex").write_text(
        "\n".join(family_lines) + "\n", encoding="utf-8",
    )
    print(json.dumps({"models": list(evidence["models"]), "items": evidence["items"]}, indent=2))


if __name__ == "__main__":
    main()
