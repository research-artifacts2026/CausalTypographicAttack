from collections import Counter
from pathlib import Path

from PIL import Image

from cta.question_bench import file_sha256
from cta.rvta_qa_balanced import (
    CONDITIONS,
    assign_counterbalance,
    build_spec,
    condition_text,
    parse_semantic_answer,
    render_condition,
    summarize,
)


def sample(tmp_path: Path, index: int = 0, label: str = "apple") -> dict:
    image = tmp_path / f"image-{index}.jpg"
    Image.new("RGB", (640, 480), (140, 170, 120)).save(image)
    return {
        "sample_id": f"sample-{index}",
        "image_path": str(image),
        "source_sha256": file_sha256(image),
        "target_label": label,
    }


def test_counterbalance_is_exact_and_order_independent(tmp_path):
    rows = [sample(tmp_path, index) for index in range(60)]
    assigned = assign_counterbalance(rows, 9)
    counts = Counter(cell for _, cell in assigned)
    assert set(counts.values()) == {10}
    assert assigned == assign_counterbalance(reversed(rows), 9)


def test_truth_and_option_order_change_correct_tokens(tmp_path):
    row = sample(tmp_path)
    false_no_yes = build_spec(row, "COCO", ("false", "ab", "no_yes"))
    false_yes_no = build_spec(row, "COCO", ("false", "ab", "yes_no"))
    true_no_yes = build_spec(row, "COCO", ("true", "ab", "no_yes"))
    assert (false_no_yes.correct_semantic, false_no_yes.correct_answer_token) == ("no", "A")
    assert (false_yes_no.correct_semantic, false_yes_no.correct_answer_token) == ("no", "B")
    assert (true_no_yes.correct_semantic, true_no_yes.correct_answer_token) == ("yes", "B")
    assert false_no_yes.verification_claim != true_no_yes.verification_claim


def test_parser_respects_registered_answer_layout():
    assert parse_semantic_answer("A", "ab", "no_yes") == "no"
    assert parse_semantic_answer("A", "ab", "yes_no") == "yes"
    assert parse_semantic_answer("B", "ab", "no_yes") == "yes"
    assert parse_semantic_answer("NO", "yesno", "semantic") == "no"
    assert parse_semantic_answer("A", "yesno", "semantic") == "invalid"


def test_bridge_targets_both_semantic_directions_without_answer_line(tmp_path):
    false_spec = build_spec(sample(tmp_path, 1), "COCO", ("false", "yesno", "semantic"))
    true_spec = build_spec(sample(tmp_path, 2), "COCO", ("true", "yesno", "semantic"))
    _, false_lines, _, _ = condition_text(false_spec, "causal_bridge")
    _, true_lines, _, _ = condition_text(true_spec, "causal_bridge")
    assert "compatible" in false_lines[1]
    assert "does not hold" in true_lines[1]
    assert "ANSWER:" not in " ".join(false_lines + true_lines)


def test_all_conditions_render_with_fixed_attack_geometry(tmp_path):
    spec = build_spec(sample(tmp_path), "COCO", ("true", "ab", "yes_no"))
    areas = []
    for condition in CONDITIONS:
        rendered = render_condition(spec, condition, tmp_path / condition / "x.jpg")
        assert Path(rendered["image_path"]).is_file()
        if condition != "no_attack":
            areas.append(rendered["overlay_area_fraction"])
    assert max(areas) - min(areas) < 1e-12


def test_summary_uses_per_item_semantic_target():
    rows = []
    for item_id, correct, target, cell in (
        ("false", "no", "yes", "false:yesno:semantic"),
        ("true", "yes", "no", "true:yesno:semantic"),
    ):
        for condition in CONDITIONS:
            parsed = correct if condition == "no_attack" else target
            rows.append({
                "item_id": item_id,
                "condition": condition,
                "parsed_semantic": parsed,
                "correct_semantic": correct,
                "target_semantic": target,
                "counterbalance_cell": cell,
                "read_match": condition != "no_attack",
            })
    summary = summarize(rows)
    pooled = {row["condition"]: row for row in summary["pooled"]}
    assert pooled["causal_bridge"]["clean_conditioned_target_asr"] == 1.0
    assert pooled["causal_bridge"]["grounded_clean_conditioned_asr"] == 1.0

