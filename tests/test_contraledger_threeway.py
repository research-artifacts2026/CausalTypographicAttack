from pathlib import Path

from PIL import Image

from cta.contraledger_threeway import (
    CONDITIONS,
    option_for_semantic,
    option_map,
    parse_choice,
    render_item,
    summarize,
)
from cta.question_bench import file_sha256
from cta.scei_attack import compile_counterfactual, fallback_scene_plan
from cta.scene_question_designer import build_scene_question
from scripts.analyze_contraledger_threeway import audit_manifest, audit_predictions


def _source_row(tmp_path: Path) -> dict:
    source = tmp_path / "oven.jpg"
    Image.new("RGB", (960, 640), (122, 142, 158)).save(source, quality=95)
    record = compile_counterfactual(
        "oven", "unit_conversion", variant_key="threeway-test", seed=20260904
    )
    plan = fallback_scene_plan("oven", "unit_conversion", "oven-3way")
    scene_question = build_scene_question(
        record, visible_object="oven", truth="false", item_id="oven-3way"
    )
    return {
        "item_id": "oven-3way",
        "dataset": "unit",
        "family": "unit_conversion",
        "scenario_id": record.scenario_id,
        "target_label": "oven",
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "record": record.to_dict(),
        "plan": plan.to_dict(),
        "scene_question": scene_question.to_dict(),
    }


def test_option_counterbalancing_and_parser():
    for index in range(6):
        mapping = option_map(index)
        assert set(mapping.values()) == {"consistent", "inconsistent", "absent"}
        for semantic in mapping.values():
            letter = option_for_semantic(mapping, semantic)
            assert parse_choice(letter, mapping) == semantic
            assert parse_choice(f"({letter})", mapping) == semantic


def test_rendered_triplet_is_frozen_and_auditable(tmp_path: Path):
    rows = render_item(_source_row(tmp_path), tmp_path / "out", permutation_index=2)
    assert [row["condition"] for row in rows] == list(CONDITIONS)
    assert len({row["question"] for row in rows}) == 1
    assert [row["correct_semantic"] for row in rows] == [
        "absent", "consistent", "inconsistent"
    ]
    assert rows[0]["image_sha256"] == rows[0]["source_sha256"]
    assert rows[1]["carrier_quad"] == rows[2]["carrier_quad"]
    assert rows[1]["mask_sha256"] == rows[2]["mask_sha256"]
    assert rows[1]["registered_read_text"] != rows[2]["registered_read_text"]
    assert audit_manifest(rows)["status"] == "passed"

    predictions = []
    for row in rows:
        prediction = {**row, "decide_parsed": row["correct_semantic"]}
        if row["condition"] == "record_false":
            prediction.update({"read_match": True, "knowledge_parsed": "no"})
        predictions.append(prediction)
    frozen = {(row["item_id"], row["condition"]): row for row in rows}
    audit_predictions("unit", predictions, frozen)


def test_summary_uses_double_control_eligibility():
    rows = []
    for item_id, attacked in (("a", True), ("b", False), ("c", True)):
        family = "unit_conversion"
        source_answer = "absent" if item_id != "c" else "consistent"
        for condition, answer in (
            ("source_absent", source_answer),
            ("record_true", "consistent"),
            ("record_false", "consistent" if attacked else "inconsistent"),
        ):
            row = {
                "item_id": item_id,
                "condition": condition,
                "family": family,
                "decide_parsed": answer,
            }
            if condition == "record_false":
                row.update({"read_match": True, "knowledge_parsed": "no"})
            rows.append(row)
    result = summarize(rows)
    assert result["items"] == 3
    assert result["n_source_and_true_correct"] == 2
    assert result["false_target_asr"] == 0.5
    assert result["false_accuracy"] == 0.5
    assert result["eor_n"] == 2
    assert result["eor_rate"] == 0.5
