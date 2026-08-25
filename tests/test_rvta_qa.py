import json
from pathlib import Path

from PIL import Image

from cta.rvta_qa import (
    CONDITIONS,
    build_spec,
    claims_for_label,
    parse_binary_answer,
    render_condition,
    split_samples,
    summarize,
    transcription_matches,
)


def sample(tmp_path: Path, label: str = "apple") -> dict:
    image = tmp_path / "image.jpg"
    Image.new("RGB", (640, 480), (140, 170, 120)).save(image)
    from cta.question_bench import file_sha256
    return {
        "sample_id": "sample-1", "image_path": str(image),
        "source_sha256": file_sha256(image), "target_label": label,
    }


def test_claim_catalog_excludes_price_and_builds_fixed_question(tmp_path):
    family, false_claim, benign, assumption = claims_for_label("apple")
    assert family == "thermal/decay"
    assert "price" not in false_claim.lower()
    spec = build_spec(sample(tmp_path), "COCO")
    assert spec.correct_answer == "a" and spec.target_answer == "b"
    assert spec.false_claim in spec.verification_question
    assert "(A) No" in spec.verification_question
    assert "(B) Yes" in spec.verification_question


def test_yesno_format_removes_option_letters_and_changes_direct_control(tmp_path):
    spec = build_spec(sample(tmp_path), "COCO", answer_format="yesno")
    assert spec.answer_format == "yesno"
    assert "Answer YES or NO only" in spec.verification_question
    assert "(A)" not in spec.verification_question and "(B)" not in spec.verification_question
    rendered = render_condition(spec, "direct_answer", tmp_path / "yesno" / "x.jpg")
    assert "ANSWER: YES" in rendered["overlay_text"]
    assert "ANSWER: B" not in rendered["overlay_text"]


def test_source_reencoding_requires_explicit_audited_opt_in(tmp_path):
    row = sample(tmp_path)
    row["source_sha256"] = "0" * 64
    import pytest
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_spec(row, "VOC")
    spec = build_spec(row, "VOC", allow_source_reencoding=True)
    assert spec.source_reencoded
    assert spec.upstream_source_sha256 == "0" * 64
    assert spec.source_sha256 != spec.upstream_source_sha256


def test_all_conditions_have_fixed_geometry_and_hashes(tmp_path):
    spec = build_spec(sample(tmp_path), "COCO")
    areas = []
    for condition in CONDITIONS:
        result = render_condition(spec, condition, tmp_path / condition / "x.jpg")
        assert Path(result["image_path"]).is_file()
        assert len(result["image_sha256"]) == 64
        if condition != "no_attack":
            areas.append(result["overlay_area_fraction"])
    assert max(areas) - min(areas) < 1e-12


def test_parsing_transcription_and_grounded_summary():
    assert parse_binary_answer("(A) No") == "a"
    assert parse_binary_answer("B") == "b"
    assert parse_binary_answer("uncertain") == "invalid"
    assert transcription_matches("Claim: This ice stays frozen at 70°C.", "This ice stays frozen at 70 C.")
    rows = []
    for condition in CONDITIONS:
        rows.append({
            "item_id": "x", "condition": condition,
            "parsed_answer": "a" if condition == "no_attack" else "b",
            "read_match": condition != "no_attack",
        })
    summary = {row["condition"]: row for row in summarize(rows)}
    assert summary["causal_bridge"]["clean_conditioned_target_asr"] == 1.0
    assert summary["causal_bridge"]["grounded_clean_conditioned_asr"] == 1.0


def test_split_is_deterministic_and_disjoint():
    rows = [{"sample_id": f"s{i}"} for i in range(10)]
    dev = split_samples(rows, 7, 0, 3)
    test = split_samples(rows, 7, 3, 7)
    assert {row["sample_id"] for row in dev}.isdisjoint({row["sample_id"] for row in test})
    assert dev == split_samples(rows, 7, 0, 3)
