#!/usr/bin/env python3
"""Aggregate audited RVTA severity logs without pooling incompatible families."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def exact_two_sided(smaller: int, total: int) -> float:
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(smaller + 1)) / (2 ** total)
    return min(1.0, 2 * tail)


def summarize_group(model: str, dimension: str, value: str, rows: list[dict]) -> dict:
    successes = sum(bool(row.get("attack_success")) for row in rows)
    transcribed = sum(bool(row.get("claim_matches_overlay")) for row in rows)
    low, high = wilson(successes, len(rows))
    return {
        "model": model,
        "dimension": dimension,
        "value": value,
        "n": len(rows),
        "strict_false_claim_acceptance_asr": successes / len(rows),
        "asr_wilson95_low": low,
        "asr_wilson95_high": high,
        "grounded_transcription_rate": transcribed / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-log", action="append", required=True, help="MODEL=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summaries: list[dict] = []
    paired_tests: list[dict] = []
    evidence = {"schema_version": "cta/violation-severity-analysis-v1", "models": {}}
    for assignment in args.model_log:
        if "=" not in assignment:
            raise ValueError("--model-log must have MODEL=PATH form")
        model, value = assignment.split("=", 1)
        path = Path(value).resolve()
        all_rows = read_jsonl(path)
        rows = [
            row for row in all_rows
            if row.get("attack_metadata", {}).get("severity")
        ]
        if not rows:
            raise ValueError(f"{model}: no severity rows")
        provenance = path.parent / "provenance.json"
        if not provenance.is_file():
            raise FileNotFoundError(provenance)
        provenance_record = json.loads(provenance.read_text(encoding="utf-8"))
        if (
            int(provenance_record.get("completed_rows", -1)) != len(all_rows)
            or int(provenance_record.get("selected_rows", -1)) != len(all_rows)
        ):
            raise ValueError(f"{model}: run provenance does not attest a complete log")
        by_dimension: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            meta = row["attack_metadata"]
            by_dimension[("severity", meta["severity"])].append(row)
            by_dimension[("family", meta["family"])].append(row)
            by_dimension[("scenario", meta["scenario_id"])].append(row)
            by_dimension[("scenario_severity", f"{meta['scenario_id']}::{meta['severity']}")].append(row)
        model_summaries = [
            summarize_group(model, dimension, value, group)
            for (dimension, value), group in sorted(by_dimension.items())
        ]
        summaries.extend(model_summaries)

        by_scenario: dict[str, dict[str, dict[str, dict]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for row in rows:
            meta = row["attack_metadata"]
            by_scenario[meta["scenario_id"]][row["sample_id"]][meta["severity"]] = row
        model_tests = []
        for scenario, by_sample in sorted(by_scenario.items()):
            pairs = [
                levels for levels in by_sample.values()
                if "moderate" in levels and "extreme" in levels
            ]
            extreme_only = sum(
                bool(levels["extreme"]["attack_success"])
                and not bool(levels["moderate"]["attack_success"])
                for levels in pairs
            )
            moderate_only = sum(
                bool(levels["moderate"]["attack_success"])
                and not bool(levels["extreme"]["attack_success"])
                for levels in pairs
            )
            discordant = extreme_only + moderate_only
            test = {
                "model": model,
                "scenario": scenario,
                "n_paired": len(pairs),
                "extreme_only_successes": extreme_only,
                "moderate_only_successes": moderate_only,
                "exact_mcnemar_p": exact_two_sided(
                    min(extreme_only, moderate_only), discordant,
                ),
            }
            paired_tests.append(test)
            model_tests.append(test)
        evidence["models"][model] = {
            "log": str(path),
            "provenance": str(provenance.resolve()),
            "summaries": model_summaries,
            "paired_moderate_vs_extreme": model_tests,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "severity_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8",
    )
    with (args.output_dir / "severity_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (args.output_dir / "severity_paired_tests.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_tests[0]))
        writer.writeheader()
        writer.writerows(paired_tests)
    print(json.dumps({"models": list(evidence["models"]), "summary_rows": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
