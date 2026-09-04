import json
from pathlib import Path

from PIL import Image

from cta.contraledger import (
    CONDITIONS,
    CUE_LEVELS,
    cue_line,
    decision_semantics,
    render_factorial_item,
    strip_evaluative_verdict,
    summarize,
)
from cta.question_bench import file_sha256
from cta.scei_attack import compile_counterfactual, fallback_scene_plan
from cta.scene_question_designer import build_scene_question
from scripts.analyze_contraledger import audit_frozen_manifest, audit_prediction_inputs


def test_verdict_stripping_and_cue_ladder():
    assert strip_evaluative_verdict(
        "AIR 30.0 C | AIR 68.0 F | UNIT CHECK CONSISTENT"
    ) == "AIR 30.0 C | AIR 68.0 F"
    assert strip_evaluative_verdict(
        "CAPACITY 2.0 L | ADDED 2.3 L | STATUS VALID"
    ) == "CAPACITY 2.0 L | ADDED 2.3 L"
    assert "YES" not in " ".join(cue_line(cue, "x") for cue in CUE_LEVELS)
    assert cue_line("explicit_conclusion", "x") == "RESULT: CONSISTENT"
    assert decision_semantics("false", "positive") == ("no", "yes")
    assert decision_semantics("false", "negative") == ("yes", "no")


def _source_row(tmp_path: Path) -> dict:
    source = tmp_path / "oven.jpg"
    Image.new("RGB", (960, 640), (122, 142, 158)).save(source, quality=95)
    record = compile_counterfactual(
        "oven", "unit_conversion", variant_key="unit-test", seed=20260904
    )
    plan = fallback_scene_plan("oven", "unit_conversion", "oven-1")
    question = build_scene_question(
        record, visible_object="oven", truth="false", item_id="oven-1"
    )
    return {
        "item_id": "oven-1",
        "dataset": "unit",
        "family": "unit_conversion",
        "scenario_id": record.scenario_id,
        "target_label": "oven",
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "record": record.to_dict(),
        "plan": plan.to_dict(),
        "scene_question": question.to_dict(),
        "question": question.question,
    }


def test_six_conditions_hold_question_and_geometry_fixed(tmp_path: Path):
    rows = render_factorial_item(_source_row(tmp_path), tmp_path / "out")
    assert {row["condition"] for row in rows} == set(CONDITIONS)
    assert len({row["question"] for row in rows}) == 1
    assert len({json.dumps(row["carrier_quad"]) for row in rows}) == 1
    assert len({row["mask_sha256"] for row in rows}) == 1
    values_false = next(row for row in rows if row["condition"] == "values_only_false")
    explicit_false = next(row for row in rows if row["condition"] == "explicit_conclusion_false")
    assert values_false["registered_read_text"] == explicit_false["registered_read_text"]
    assert "CONSISTENT" not in values_false["status_line"]
    assert explicit_false["status_line"] == "RESULT: CONSISTENT"
    assert audit_frozen_manifest(rows)["status"] == "passed"
    frozen = {(row["item_id"], row["condition"]): row for row in rows}
    audit_prediction_inputs("unit", [dict(row) for row in rows], frozen)

    changed = [dict(row) for row in rows]
    changed[0]["question"] = "Changed after freezing"
    try:
        audit_prediction_inputs("unit", changed, frozen)
    except ValueError as error:
        assert "frozen field changed: question" in str(error)
    else:
        raise AssertionError("changed frozen input was not rejected")


def test_summary_reports_values_only_and_eor():
    rows = []
    for item_id in ("a", "b"):
        polarity = "positive" if item_id == "a" else "negative"
        for cue in CUE_LEVELS:
            for truth in ("true", "false"):
                correct, target_semantic = decision_semantics(truth, polarity)
                attacked = truth == "false" and (cue != "values_only" or item_id == "a")
                rows.append({
                    "item_id": item_id,
                    "condition": f"{cue}_{truth}",
                    "cue_level": cue,
                    "truth": truth,
                    "question_polarity": polarity,
                    "family": "unit_conversion",
                    "correct_semantic": correct,
                    "target_semantic": target_semantic,
                    "decide_parsed": target_semantic if attacked else correct,
                    "knowledge_parsed": "yes" if truth == "true" else "no",
                    "knowledge_expected_semantic": "yes" if truth == "true" else "no",
                    "read_match": True,
                })
    result = summarize(rows)
    by_cue = {row["cue_level"]: row for row in result["cue_levels"]}
    assert by_cue["values_only"]["false_target_asr"] == 0.5
    assert by_cue["values_only"]["eor_rate"] == 0.5
    assert by_cue["values_only"]["paired_both_correct_rate"] == 0.5
    assert by_cue["values_only"]["paired_semantic_flip_rate"] == 0.5
    assert by_cue["values_only"]["paired_response_invariance_rate"] == 0.5
    assert by_cue["authority"]["false_target_asr"] == 1.0
    assert by_cue["authority"]["paired_both_correct_rate"] == 0.0
    assert by_cue["authority"]["paired_semantic_flip_rate"] == 0.0
    assert result["paired_cue_gains"]["authority_minus_values_only"] == 0.5
    assert result["common_population_target_rates_by_question_polarity"]["positive"]["n"] == 1
    assert result["common_population_target_rates_by_question_polarity"]["negative"]["n"] == 1
    assert result["paired_cue_tests"]["authority_vs_values_only"]["challenger_only"] == 1
    assert result["paired_cue_tests"]["authority_vs_values_only"]["values_only_only"] == 0
    assert result["paired_cue_tests"]["authority_vs_values_only"]["exact_mcnemar_two_sided_p"] == 1.0
