#!/usr/bin/env python3
"""Validate and summarize completed independent human ratings."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from make_human_eval_pack import RATING_COLUMNS


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position); upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def clustered_bootstrap_ci(rows: list[dict], column: str, seed: int, draws: int = 10000) -> tuple[float, float]:
    """Bootstrap matched image identifiers, retaining every annotator rating in each draw."""
    rng = random.Random(seed)
    by_sample: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(row[column])
    sample_ids = sorted(by_sample)
    means = []
    for _ in range(draws):
        sampled = [rng.choice(sample_ids) for _ in sample_ids]
        means.append(statistics.fmean(value for sample_id in sampled for value in by_sample[sample_id]))
    return percentile(means, 0.025), percentile(means, 0.975)


def paired_clustered_bootstrap_difference(
    rows: list[dict], left: str, right: str, column: str, seed: int, draws: int = 10000,
) -> dict:
    """Return right-minus-left mean difference with an image-clustered percentile interval."""
    by_method_sample: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_method_sample[row["method"]][row["sample_id"]].append(row[column])
    sample_ids = sorted(set(by_method_sample[left]) & set(by_method_sample[right]))
    if not sample_ids:
        raise ValueError(f"no matched sample ids for {left} versus {right}")
    deltas = {
        sample_id: statistics.fmean(by_method_sample[right][sample_id])
        - statistics.fmean(by_method_sample[left][sample_id])
        for sample_id in sample_ids
    }
    rng = random.Random(seed)
    draws_values = [
        statistics.fmean(deltas[rng.choice(sample_ids)] for _ in sample_ids)
        for _ in range(draws)
    ]
    return {
        "left": left,
        "right": right,
        "n_matched_images": len(sample_ids),
        "mean_difference_right_minus_left": statistics.fmean(deltas.values()),
        "ci95": [percentile(draws_values, 0.025), percentile(draws_values, 0.975)],
    }


def krippendorff_interval(item_values: dict[str, list[float]]) -> float | None:
    observed_terms = []
    all_values = []
    for values in item_values.values():
        all_values.extend(values)
        if len(values) > 1:
            observed_terms.extend((a - b) ** 2 for i, a in enumerate(values) for b in values[i + 1:])
    if not observed_terms or len(all_values) < 2:
        return None
    expected_terms = [(a - b) ** 2 for i, a in enumerate(all_values) for b in all_values[i + 1:]]
    expected = statistics.fmean(expected_terms)
    return None if expected == 0 else 1.0 - statistics.fmean(observed_terms) / expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--minimum-annotators", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260312)
    parser.add_argument(
        "--responses-dir", default="responses",
        help="response directory relative to the pack root",
    )
    parser.add_argument(
        "--output", default="human_results.json",
        help="result filename relative to the pack root",
    )
    parser.add_argument(
        "--evaluator-kind", choices=("human", "model"), default="human",
        help="evidence label; model ratings must never be reported as human ratings",
    )
    parser.add_argument(
        "--evaluator-model", default=None,
        help="model identifier required when --evaluator-kind=model",
    )
    args = parser.parse_args()
    if args.evaluator_kind == "model" and not args.evaluator_model:
        raise ValueError("--evaluator-model is required for model evaluation")
    response_paths = sorted((args.pack_root / args.responses_dir).glob("*.csv"))
    if len(response_paths) < args.minimum_annotators:
        raise ValueError(f"need at least {args.minimum_annotators} independent response files; found {len(response_paths)}")
    method_key = {row["item_id"]: row for row in read_csv(args.pack_root / "private_method_key.csv")}
    ratings = []
    per_annotator_items: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for response_path in response_paths:
        annotator = response_path.stem
        for row in read_csv(response_path):
            if row["item_id"] not in method_key:
                raise ValueError(f"unknown item id in {response_path}: {row['item_id']}")
            for column in RATING_COLUMNS:
                try:
                    value = float(row[column])
                except ValueError as exc:
                    raise ValueError(f"missing/non-numeric {column} in {response_path}, row {row['row_id']}") from exc
                if not 1 <= value <= 5:
                    raise ValueError(f"out-of-range {column} in {response_path}, row {row['row_id']}")
                row[column] = value
            row["annotator"] = annotator
            row.update(method_key[row["item_id"]])
            ratings.append(row)
            per_annotator_items[annotator][row["item_id"]].append(row)

    duplicate_mae = []
    deduplicated = []
    for by_item in per_annotator_items.values():
        for copies in by_item.values():
            deduplicated.append(copies[0])
            if len(copies) > 1:
                for column in RATING_COLUMNS:
                    duplicate_mae.append(abs(copies[0][column] - copies[1][column]))
    item_annotators: dict[str, set[str]] = defaultdict(set)
    for row in deduplicated:
        item_annotators[row["item_id"]].add(row["annotator"])
    incomplete = [item for item, annotators in item_annotators.items() if len(annotators) < args.minimum_annotators]
    if incomplete:
        raise ValueError(f"{len(incomplete)} items have fewer than {args.minimum_annotators} independent ratings")

    result = {"evaluators": len(response_paths), "evaluator_kind": args.evaluator_kind,
              "evaluator_model": args.evaluator_model,
              "evidence_label": (
                  "independent blinded model-evaluation runs"
                  if args.evaluator_kind == "model" else "independent blinded human annotators"
              ),
              "ratings_after_dedup": len(deduplicated),
              "duplicate_mean_absolute_difference": statistics.fmean(duplicate_mae) if duplicate_mae else None,
              "inference": {"unit": "matched image identifier", "bootstrap_draws": 10000, "seed": args.seed},
              "methods": {}}
    for method in sorted({row["method"] for row in deduplicated}):
        method_rows = [row for row in deduplicated if row["method"] == method]
        result["methods"][method] = {}
        for metric_index, column in enumerate(RATING_COLUMNS):
            values = [row[column] for row in method_rows]
            low, high = clustered_bootstrap_ci(method_rows, column, args.seed + metric_index)
            result["methods"][method][column] = {
                "n_ratings": len(values), "mean": statistics.fmean(values), "ci95": [low, high],
            }
    result["paired_differences"] = {}
    methods = sorted(result["methods"])
    for metric_index, column in enumerate(RATING_COLUMNS):
        result["paired_differences"][column] = [
            paired_clustered_bootstrap_difference(
                deduplicated, left, right, column,
                args.seed + 100 + 10 * metric_index + pair_index,
            )
            for pair_index, (left, right) in enumerate(combinations(methods, 2))
        ]
    reliability = {}
    for column in RATING_COLUMNS:
        by_item = defaultdict(list)
        for row in deduplicated:
            by_item[row["item_id"]].append(row[column])
        reliability[column] = krippendorff_interval(by_item)
    result["krippendorff_alpha_interval"] = reliability
    (args.pack_root / args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
