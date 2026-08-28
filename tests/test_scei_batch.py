from __future__ import annotations

from cta.scei_batch import (
    assign_family_stratified_splits,
    safe_item_slug,
    select_family_balanced,
    select_requested_families,
    summarize_terminal_rows,
    wilson_interval,
)
from cta.scei_attack import (
    REQUESTED_COUNTERFACTUAL_FAMILIES,
    compile_counterfactual,
    recompute_record_residual,
    validate_record,
)


def _row(item: int, *, clean: bool, s1: int, s2: int, outcome: str, feedback: list[str]) -> dict:
    return {
        "item_id": f"item-{item}",
        "clean_correct": clean,
        "success_at_1": s1,
        "success_at_2": s2,
        "victim_query_count": 1 if not clean else (3 if s1 else 5),
        "planner_query_count": 0 if not clean else (1 if s1 else 2),
        "victim_queries_to_success": 3 if s1 else (5 if s2 else None),
        "terminal_outcome": outcome,
        "round_feedback": feedback,
        "family": "mass_balance",
    }


def test_summary_is_clean_conditioned_but_keeps_clean_errors() -> None:
    rows = [
        _row(1, clean=False, s1=0, s2=0, outcome="clean_error", feedback=[]),
        _row(2, clean=True, s1=1, s2=1, outcome="strict_success_round_1", feedback=["strict_success"]),
        _row(3, clean=True, s1=0, s2=1, outcome="strict_success_round_2", feedback=["read_but_resisted", "strict_success"]),
        _row(4, clean=True, s1=0, s2=0, outcome="budget_not_read_or_partial", feedback=["not_read_or_partial"]),
    ]
    summary = summarize_terminal_rows(rows, expected_items=4, max_rounds=2)
    assert summary["status"] == "complete"
    assert summary["clean_correct"] == 3
    assert summary["clean_errors"] == 1
    assert summary["success_at_1"]["successes"] == 1
    assert summary["success_at_1"]["denominator_clean_correct"] == 3
    assert summary["success_at_2"]["successes"] == 2
    assert summary["terminal_outcomes"]["clean_error"] == 1
    family = summary["family_metrics"]["mass_balance"]
    assert family["n_clean_correct"] == 3
    assert family["strict_success_at_1"] == 1
    assert family["strict_success_at_2"] == 2
    assert family["target_flip_at_1"] == 1
    assert family["exact_read_at_2"] == 2


def test_wilson_and_safe_slug_are_stable() -> None:
    lower, upper = wilson_interval(5, 10)
    assert 0.23 < lower < 0.24
    assert 0.76 < upper < 0.77
    assert safe_item_slug("a/b:c") == safe_item_slug("a/b:c")
    assert "/" not in safe_item_slug("a/b:c")


def test_selection_is_deterministic_and_offset_disjoint() -> None:
    rows = [
        {"sample_id": f"sample-{index:03d}", "target_label": "pizza", "image_path": "unused"}
        for index in range(100)
    ]
    development = select_family_balanced(rows, seed=20260827, offset=0, limit=48)
    evaluation = select_family_balanced(rows, seed=20260827, offset=48, limit=50)
    repeated = select_family_balanced(rows, seed=20260827, offset=48, limit=50)
    assert evaluation == repeated
    assert {row["sample_id"] for row in development}.isdisjoint(
        {row["sample_id"] for row in evaluation}
    )


def test_named_family_compiler_has_one_field_twins_and_independent_truth_checks() -> None:
    for family in REQUESTED_COUNTERFACTUAL_FAMILIES:
        record = compile_counterfactual("truck", family)
        validate_record(record)
        false_parts = [part.strip() for part in record.false_measurement.split("|")]
        true_parts = [part.strip() for part in record.true_measurement.split("|")]
        assert len(false_parts) == len(true_parts)
        assert sum(left != right for left, right in zip(false_parts, true_parts)) == 1
        assert abs(recompute_record_residual(record, "false")) > record.tolerance
        assert abs(recompute_record_residual(record, "true")) <= record.tolerance


def test_requested_suite_is_balanced_and_excludes_first_48_development_items() -> None:
    labels = ["truck", "bowl", "refrigerator", "bench", "person"]
    rows = [
        {
            "sample_id": f"sample-{index:03d}",
            "target_label": labels[index % len(labels)],
            "labels": [labels[index % len(labels)]],
            "image_path": "unused",
        }
        for index in range(160)
    ]
    development = select_family_balanced(rows, seed=20260827, offset=0, limit=48)
    evaluation = select_requested_families(
        rows, seed=20260827, development_offset=48, limit=50
    )
    assert {row["sample_id"] for row in development}.isdisjoint(
        {row["sample_id"] for row in evaluation}
    )
    counts = {family: 0 for family in REQUESTED_COUNTERFACTUAL_FAMILIES}
    for row in evaluation:
        counts[row["family"]] += 1
    assert sorted(counts.values()) == [6, 6, 6, 6, 6, 6, 7, 7]


def test_v2_compiler_is_deterministic_diverse_and_mechanically_valid() -> None:
    signatures: set[tuple[str, str, str]] = set()
    for family in REQUESTED_COUNTERFACTUAL_FAMILIES:
        family_signatures = set()
        for index in range(30):
            key = f"item-{index:04d}"
            first = compile_counterfactual("truck", family, variant_key=key, seed=20260828)
            repeated = compile_counterfactual("truck", family, variant_key=key, seed=20260828)
            assert first == repeated
            validate_record(first)
            false_parts = [part.strip() for part in first.false_measurement.split("|")]
            true_parts = [part.strip() for part in first.true_measurement.split("|")]
            assert sum(left != right for left, right in zip(false_parts, true_parts)) == 1
            assert first.generator_version == "scei-symbolic-v2"
            assert first.difficulty in {"subtle", "moderate", "strong"}
            assert first.changed_field
            signature = (family, first.false_measurement, first.true_measurement)
            signatures.add(signature)
            family_signatures.add(signature)
        assert len(family_signatures) >= 8
    assert len(signatures) >= 100


def test_v2_compiler_can_allocate_one_hundred_unique_records_per_family() -> None:
    for family in REQUESTED_COUNTERFACTUAL_FAMILIES:
        signatures: set[tuple[str, str]] = set()
        for index in range(100):
            base_key = f"{family}-{index:03d}"
            for collision_index in range(100):
                key = base_key if collision_index == 0 else f"{base_key}:collision-{collision_index}"
                record = compile_counterfactual("object", family, variant_key=key, seed=20260828)
                validate_record(record)
                signature = (record.false_measurement, record.true_measurement)
                if signature not in signatures:
                    signatures.add(signature)
                    break
            else:
                raise AssertionError(f"{family}: exhausted collision budget at item {index}")
        assert len(signatures) == 100


def test_family_stratified_splits_are_exact_and_deterministic() -> None:
    rows = [
        {"item_id": f"{family}-{index:03d}", "family": family}
        for family in REQUESTED_COUNTERFACTUAL_FAMILIES
        for index in range(100)
    ]
    counts = {"train": 70, "validation": 15, "test": 15}
    first = assign_family_stratified_splits(rows, split_counts_per_family=counts, seed=20260828)
    second = assign_family_stratified_splits(rows, split_counts_per_family=counts, seed=20260828)
    assert first == second
    for family in REQUESTED_COUNTERFACTUAL_FAMILIES:
        subset = [row for row in first if row["family"] == family]
        assert {split: sum(row["split"] == split for row in subset) for split in counts} == counts
