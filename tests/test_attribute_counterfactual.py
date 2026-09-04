import json
from pathlib import Path

from PIL import Image

from cta.attribute_counterfactual import (
    CONDITIONS,
    FAMILIES,
    build_item,
    exact_read_match,
    probe_prompts,
    render_condition,
    select_family_items,
    summarize,
)
from cta.question_bench import file_sha256


def _image(path: Path) -> None:
    Image.new("RGB", (960, 640), (128, 146, 164)).save(path, quality=95)


def test_family_targets_and_catalog_are_balanced():
    targets = [spec.false_decision_semantic for spec in FAMILIES.values()]
    assert targets.count("yes") == targets.count("no") == 3
    assert set(FAMILIES) == {"value", "weight", "temperature", "capacity", "age", "energy"}


def test_family_selection_uses_unique_compatible_scenes():
    rows = []
    label_by_family = {
        "value": "chair",
        "weight": "book",
        "temperature": "cup",
        "capacity": "bottle",
        "age": "bus",
        "energy": "oven",
    }
    for family, label in label_by_family.items():
        for index in range(3):
            rows.append({
                "item_id": f"{family}-{index}",
                "target_label": label,
                "visible_labels": [label],
                "plan": {"scene_description": f"A clearly visible {label} in the scene."},
            })
    selected = select_family_items(rows, 3, 11)
    assert len(selected) == 18
    assert len({row["item_id"] for _, row in selected}) == 18


def test_selection_reroutes_to_a_scene_described_visible_object():
    rows = []
    label_by_family = {
        "value": "chair",
        "weight": "book",
        "temperature": "cup",
        "capacity": "bottle",
        "age": "bus",
        "energy": "oven",
    }
    for family, label in label_by_family.items():
        for index in range(2):
            rows.append({
                "item_id": f"reroute-{family}-{index}",
                "target_label": "person",
                "visible_labels": ["person", label],
                "plan": {"scene_description": f"A person stands beside a prominent {label}."},
            })
    selected = select_family_items(rows, 2, 19)
    assert len(selected) == 12
    for family, row in selected:
        assert row["attribute_cf_target_label"] in FAMILIES[family].compatible_labels
        assert row["attribute_cf_selection_evidence"] == "clean_scene_description"


def test_selection_rejects_incidental_annotation_not_in_clean_description():
    rows = [{
        "item_id": "tiny-bottle",
        "target_label": "bottle",
        "visible_labels": ["bottle", "person"],
        "plan": {"scene_description": "A person stands in a town square."},
    }]
    try:
        select_family_items(rows, 1, 23)
    except ValueError as exc:
        assert "capacity" in str(exc)
    else:
        raise AssertionError("incidental target annotation should not be selected")


def test_minimal_twins_and_factorial_geometry(tmp_path: Path):
    source = tmp_path / "source.jpg"
    _image(source)
    row = {
        "item_id": "phone-1",
        "target_label": "cell phone",
        "source_path": str(source),
        "source_sha256": file_sha256(source),
    }
    item = build_item("value", row, "unit", 7, 0)
    renders = {
        condition: render_condition(item, condition, tmp_path / f"{condition}.jpg")
        for condition in CONDITIONS
    }
    boxes = {tuple(renders[condition]["bbox"]) for condition in CONDITIONS if condition != "clean"}
    assert len(boxes) == 1
    assert renders["true_plain"]["registered_read_text"] != renders["false_plain"]["registered_read_text"]
    assert renders["true_target_conclusion"]["overlay_text"].endswith(item.target_conclusion)
    assert renders["false_target_conclusion"]["overlay_text"].endswith(item.target_conclusion)
    assert "1,000,000" not in item.question
    assert "ANSWER" not in item.target_conclusion


def test_probe_prompts_are_independent_and_low_overlap(tmp_path: Path):
    source = tmp_path / "source.jpg"
    _image(source)
    row = {"item_id": "cup-1", "target_label": "cup", "source_path": str(source)}
    item = build_item("temperature", row, "unit", 3, 0)
    prompts = probe_prompts(item, "false_plain")
    assert set(prompts) == {"read", "ground", "verify", "decide"}
    assert item.false_value not in prompts["decide"]
    assert item.target_conclusion not in prompts["decide"]
    assert exact_read_match("CORE TEMPERATURE 500 C", "CORE TEMPERATURE: 500 C")
    assert not exact_read_match("The temperature is 500 C", "CORE TEMPERATURE: 500 C")


def test_summary_reports_counterfactual_effect_and_kdi():
    rows = []
    for item_id in ("a", "b"):
        for condition in CONDITIONS:
            target = "yes"
            true_answer = "no"
            decide = true_answer
            if condition in {"false_plain", "false_target_conclusion"}:
                decide = target
            rows.append({
                "item_id": item_id,
                "family": "value",
                "condition": condition,
                "true_decision_semantic": true_answer,
                "target_semantic": target,
                "decide_parsed": decide,
                "read_match": condition != "clean",
                "ground_parsed": "no" if condition == "clean" else "yes",
                "ground_expected": "no" if condition == "clean" else "yes",
                "verify_parsed": "no" if condition in {"clean", "false_plain", "false_target_conclusion"} else "yes",
                "verify_expected": "no" if condition in {"clean", "false_plain", "false_target_conclusion"} else "yes",
            })
    result = summarize(rows)
    assert result["n_common_clean_correct"] == 2
    assert result["factorial"]["counterfactual_effect_without_conclusion"] == 1.0
    false_plain = next(row for row in result["conditions"] if row["condition"] == "false_plain")
    assert false_plain["kdi_n"] == 2
    assert false_plain["kdi_rate"] == 1.0
