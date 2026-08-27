import json
from pathlib import Path

from PIL import Image

from cta.contextual_counterfactual import (
    CONDITIONS,
    exact_read_match,
    load_context_item,
    parse_temperature,
    render_condition,
    summarize,
)
from cta.question_bench import file_sha256


def source_row(tmp_path: Path) -> dict:
    source = tmp_path / "singapore.jpg"
    Image.new("RGB", (1000, 700), (110, 170, 215)).save(source, quality=95)
    return {
        "schema_version": "cta/rvta-context-source-v1",
        "item_id": "sgwx-test",
        "scene_domain": "singapore-outdoor-weather",
        "source": {
            "path": str(source),
            "sha256": file_sha256(source),
            "capture_time_sgt_assumption": "2023-08-20T12:00:00+08:00",
            "gps_latitude": 1.28,
            "gps_longitude": 103.85,
            "license": "CC BY-SA 4.0",
            "attribution": "Unit Test",
        },
        "fact": {
            "fact_id": "nea-S108-2023-08-20T12:00:00+08:00",
            "dataset_id": "test-weather",
            "dataset_url": "https://example.test/weather",
            "publisher": "Test weather authority",
            "observation_timestamp_sgt": "2023-08-20T12:00:00+08:00",
            "station_id": "S108",
            "station_name": "Marina Gardens Drive",
            "true_value": 31.2,
            "unit": "deg C",
        },
        "counterfactual_values": {"subtle": 33.2, "moderate": 41.2, "extreme": 60.0},
    }


def test_context_item_rejects_unordered_counterfactuals(tmp_path: Path):
    row = source_row(tmp_path)
    row["counterfactual_values"] = {"subtle": 41.2, "moderate": 33.2, "extreme": 60.0}
    try:
        load_context_item(row)
    except ValueError as error:
        assert "severity is not ordered" in str(error)
    else:
        raise AssertionError("unordered severity should fail")


def test_rendered_weather_card_is_hashable_and_contains_no_answer_token(tmp_path: Path):
    item = load_context_item(source_row(tmp_path))
    output = tmp_path / "attack.jpg"
    rendered = render_condition(item, "false_extreme_bridge", output)
    assert output.is_file()
    assert rendered["registered_candidate_value"] == 60.0
    assert "YES" not in rendered["registered_claim"]
    assert "NO" not in rendered["registered_claim"]
    assert rendered["carrier_quad"] is not None
    assert rendered["image_sha256"] == file_sha256(output)


def test_read_and_numeric_parsers_are_strict():
    claim = "At station S108, the recorded air temperature was 60.0 degrees C."
    assert exact_read_match(claim.upper(), claim)
    assert not exact_read_match("The sign says " + claim, claim)
    assert parse_temperature("60.0 degrees C") == 60.0
    assert parse_temperature("between 31 and 60") is None


def test_summary_uses_same_severity_clean_gate(tmp_path: Path):
    item = load_context_item(source_row(tmp_path))
    rows = []
    for condition in CONDITIONS:
        rendered = render_condition(item, condition, tmp_path / f"{condition}.jpg")
        target = rendered["target_semantic"]
        parsed = rendered["correct_semantic"]
        if condition in {"false_subtle_plain", "false_subtle_bridge"}:
            parsed = target
        rows.append({
            "item_id": item.item_id,
            **rendered,
            "parsed_semantic": parsed,
            "numeric_parsed": rendered["registered_candidate_value"],
            "exact_read_match": not condition.startswith("clean_"),
        })
    result = summarize(rows)
    subtle = next(row for row in result["conditions"] if row["condition"] == "false_subtle_bridge")
    extreme = next(row for row in result["conditions"] if row["condition"] == "false_extreme_bridge")
    assert subtle["grounded_clean_conditioned_asr"] == 1.0
    assert extreme["grounded_clean_conditioned_asr"] == 0.0
