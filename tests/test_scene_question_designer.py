from __future__ import annotations

import re

from cta.scene_question_designer import build_scene_question
from cta.scei_attack import compile_counterfactual, validate_record
from cta.scei_reasoning_families import FAMILY_IDS
from scripts.build_scene_question_manifest import build_question_rows


LABELS = {
    "range_threshold": "person",
    "unit_conversion": "car",
    "temporal_ledger": "person",
    "capacity_conservation": "bottle",
    "causal_order": "car",
    "geometry_feasibility": "suitcase",
    "probability_ledger": "book",
    "phase_state": "oven",
}


def test_all_families_produce_identical_image_dependent_twin_questions() -> None:
    for family in FAMILY_IDS:
        label = LABELS[family]
        record = compile_counterfactual(
            label, family, variant_key=f"test-{family}", seed=17,
        )
        validate_record(record)
        false = build_scene_question(
            record, visible_object=label, truth="false", item_id=family,
        )
        true = build_scene_question(
            record, visible_object=label, truth="true", item_id=family,
        )
        assert false.question == true.question
        assert false.correct_semantic == "no"
        assert true.correct_semantic == "yes"
        assert false.attack_target_semantic == "yes"
        assert true.attack_target_semantic is None
        assert label in false.question
        assert false.scene_role in false.question
        assert " a event-" not in false.question
        assert false.required_record_fields
        assert false.mechanical_rule
        measurement_numbers = set(re.findall(r"\d+(?:\.\d+)?", record.false_measurement))
        assert not measurement_numbers.intersection(re.findall(r"\d+(?:\.\d+)?", false.question_stem))


def test_question_builder_pairs_existing_manifest_rows() -> None:
    family = "capacity_conservation"
    record = compile_counterfactual("bottle", family, variant_key="pair", seed=9)
    base = {
        "item_id": "sample-1",
        "family": family,
        "target_label": "bottle",
        "record": record.to_dict(),
        "image_path": "/image.jpg",
        "carrier_quad": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "mask_sha256": "fixed-mask",
    }
    rows = build_question_rows([
        {**base, "variant": "clean"},
        {**base, "variant": "attack_false"},
        {**base, "variant": "control_true"},
    ])
    assert len(rows) == 2
    assert rows[0]["question"] == rows[1]["question"]
    assert {row["correct_semantic"] for row in rows} == {"yes", "no"}
    assert next(row for row in rows if row["variant"] == "attack_false")[
        "registered_attack_target"
    ] in {"A", "B"}
