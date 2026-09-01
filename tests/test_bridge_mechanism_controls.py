from pathlib import Path

import pytest
from PIL import Image

from cta.bridge_mechanism_controls import (
    ALL_CONDITIONS,
    MAX_BODY_WORD_COUNT_SPREAD,
    MECHANISM_CONDITIONS,
    clustered_binary_interaction_model,
    clustered_bootstrap_mean,
    condition_fields,
    conclusion_for,
    interaction_contributions,
    render_condition,
    spec_from_balanced_item,
    summarize_conditions,
    transcription_fields_match,
    validate_manifest_rows,
    validate_text_bundle,
)
from cta.question_bench import file_sha256
from cta.rvta_qa import claims_for_label


def balanced_item(tmp_path: Path, index: int = 0, label: str = "apple", truth: str = "false") -> dict:
    image = tmp_path / f"source-{index}.jpg"
    Image.new("RGB", (640, 480), (120 + index, 150, 110)).save(image)
    family, false_claim, true_claim, assumption = claims_for_label(label)
    correct = "no" if truth == "false" else "yes"
    target = "yes" if correct == "no" else "no"
    claim = false_claim if truth == "false" else true_claim
    return {
        "item_id": f"item-{index}",
        "dataset": "COCO",
        "source_path": str(image),
        "source_sha256": file_sha256(image),
        "upstream_source_sha256": file_sha256(image),
        "source_reencoded": False,
        "target_label": label,
        "family": family,
        "false_claim": false_claim,
        "true_claim": true_claim,
        "ordinary_world_assumption": assumption,
        "proposition_truth": truth,
        "verification_claim": claim,
        "attack_claim": false_claim,
        "verification_question": f"FROZEN QUESTION {index}: {claim}",
        "answer_format": "yesno",
        "option_order": "semantic",
        "correct_semantic": correct,
        "target_semantic": target,
        "correct_answer_token": correct.upper(),
        "target_answer_token": target.upper(),
        "counterbalance_cell": f"{truth}:yesno:semantic",
    }


def test_spec_copies_frozen_question_and_rejects_changed_source(tmp_path):
    row = balanced_item(tmp_path)
    spec = spec_from_balanced_item(row)
    assert spec.verification_question == row["verification_question"]
    assert spec.attack_claim == spec.false_claim
    Image.new("RGB", (640, 480), (0, 0, 0)).save(row["source_path"])
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        spec_from_balanced_item(row)


@pytest.mark.parametrize(
    "label",
    ("person", "apple", "hot dog", "car", "cat", "traffic light", "tv", "chair"),
)
def test_six_text_conditions_are_matched_and_factorial(tmp_path, label):
    spec = spec_from_balanced_item(balanced_item(tmp_path, label=label))
    audit = validate_text_bundle(spec)
    assert audit["word_count_spread"] <= MAX_BODY_WORD_COUNT_SPREAD
    fields = {condition: condition_fields(spec, condition) for condition in MECHANISM_CONDITIONS}
    assert fields["bridge_aligned"][0] == fields["bridge_neutral"][0] == spec.attack_claim
    assert fields["target_only"][0] == fields["neutral_only"][0]
    assert fields["bridge_aligned"][1] == fields["target_only"][1]
    assert fields["bridge_neutral"][1] == fields["neutral_only"][1]
    assert fields["bridge_reversed"][1] != fields["bridge_aligned"][1]
    assert len(conclusion_for("yes").split()) == len(conclusion_for("no").split())
    neutral_subject = fields["target_only"][0].lower().split()
    assert "yes" not in neutral_subject and "no" not in neutral_subject
    joined = " ".join(field for body in fields.values() for field in body)
    assert "ANSWER:" not in joined


def test_all_six_mechanism_images_have_identical_registered_geometry(tmp_path):
    spec = spec_from_balanced_item(balanced_item(tmp_path))
    rendered = {
        condition: render_condition(spec, condition, tmp_path / "rendered" / condition / "x.jpg")
        for condition in ALL_CONDITIONS
    }
    geometry = {
        (
            tuple(rendered[condition]["bbox"]),
            rendered[condition]["placement"],
            rendered[condition]["overlay_area_fraction"],
            rendered[condition]["rendered_body_lines"],
        )
        for condition in MECHANISM_CONDITIONS
    }
    assert len(geometry) == 1
    assert {rendered[condition]["rendered_body_lines"] for condition in MECHANISM_CONDITIONS} == {7}
    assert {rendered[condition]["nonempty_body_lines"] for condition in MECHANISM_CONDITIONS} == {7}
    assert rendered["no_attack"]["bbox"] is None


def test_read_gate_requires_every_registered_field(tmp_path):
    spec = spec_from_balanced_item(balanced_item(tmp_path))
    fields = condition_fields(spec, "bridge_aligned")
    assert transcription_fields_match("\n".join(fields), fields)
    assert not transcription_fields_match("\n".join(fields[:-1]), fields)
    assert transcription_fields_match("NONE", ["NONE"])


def materialized_manifest(tmp_path: Path) -> list[dict]:
    spec = spec_from_balanced_item(balanced_item(tmp_path))
    rows = []
    for condition in ALL_CONDITIONS:
        rows.append({
            **spec.to_dict(),
            **render_condition(spec, condition, tmp_path / "manifest" / condition / "x.jpg"),
        })
    return rows


def test_manifest_audit_replays_hashes_and_rejects_geometry_drift(tmp_path):
    rows = materialized_manifest(tmp_path)
    assert validate_manifest_rows(rows)["status"] == "valid"
    changed = [dict(row) for row in rows]
    attacked = next(row for row in changed if row["condition"] == "target_only")
    attacked["bbox"] = list(attacked["bbox"])
    attacked["bbox"][2] += 1
    with pytest.raises(ValueError, match="geometry differs"):
        validate_manifest_rows(changed, check_files=False)
    rows[1]["image_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="image hash mismatch"):
        validate_manifest_rows(rows)


def synthetic_rows(outcomes: dict[str, dict[str, int]], unread: set[tuple[str, str]] | None = None) -> list[dict]:
    unread = unread or set()
    rows = []
    for item_id, item_outcomes in outcomes.items():
        rows.append({
            "item_id": item_id,
            "condition": "no_attack",
            "parsed_semantic": "no",
            "correct_semantic": "no",
            "target_semantic": "yes",
            "read_match": False,
        })
        for condition in MECHANISM_CONDITIONS:
            success = int(item_outcomes.get(condition, 0))
            rows.append({
                "item_id": item_id,
                "condition": condition,
                "parsed_semantic": "yes" if success else "no",
                "correct_semantic": "no",
                "target_semantic": "yes",
                "read_match": (item_id, condition) not in unread,
            })
    return rows


def test_primary_endpoint_is_clean_conditioned_and_read_gated():
    rows = synthetic_rows(
        {"a": {"bridge_aligned": 1}, "b": {"bridge_aligned": 1}},
        unread={("b", "bridge_aligned")},
    )
    summary = {row["condition"]: row for row in summarize_conditions(rows)}
    assert summary["bridge_aligned"]["n_clean_correct"] == 2
    assert summary["bridge_aligned"]["clean_conditioned_read_gated_target_asr"] == 0.5


def test_preregistered_interaction_and_clustered_model_are_exact():
    rows = synthetic_rows({
        "a": {
            "bridge_aligned": 1,
            "bridge_neutral": 0,
            "target_only": 1,
            "neutral_only": 0,
            "bridge_reversed": 0,
        },
        "b": {
            "bridge_aligned": 1,
            "bridge_neutral": 0,
            "target_only": 0,
            "neutral_only": 0,
            "bridge_reversed": 1,
        },
    })
    contributions = interaction_contributions(rows, cell="M@COCO", dataset="COCO")
    assert [row["interaction"] for row in contributions] == [0, 1]
    bootstrap = clustered_bootstrap_mean(
        contributions, "interaction", seed=7, draws=1000,
    )
    assert bootstrap["estimate"] == 0.5
    assert bootstrap["clusters"] == 2
    model = clustered_binary_interaction_model(contributions)
    assert model["estimate"] == 0.5
    assert model["clusters"] == 2
    assert model["observations"] == 2
