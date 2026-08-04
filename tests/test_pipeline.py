from cta.generation import AttackTextGenerator, extract_json
from cta.metrics import claim_matches_overlay, label_match, parse_task_output, summarize


def test_attack_semantics():
    texts = AttackTextGenerator(42).generate("car", "abc")
    assert "car" in texts["causal"].text.lower()
    assert texts["causal"].violation_type
    assert texts["naive"].target_wrong_label != "car"


def test_json_and_output_parsing():
    assert extract_json("```json\n{\"objects\":[\"car\"]}\n```")["objects"] == ["car"]
    parsed = parse_task_output('answer: {"object":"car","claim":"FALSE"}')
    assert parsed == {"object": "car", "claim_text": "NONE", "claim": "FALSE", "parse_ok": True}
    assert label_match("cars", "car")
    assert claim_matches_overlay("This car traveled 5,000 km after one second of fueling.", "This car traveled 5,000 km after one second of fueling.")
    assert not claim_matches_overlay("car", "This car traveled 5,000 km after one second of fueling.")


def test_summary_claim_asr():
    rows = [{"attack": "causal", "defense": "none", "object_correct": True, "attack_success": True,
             "parsed": {"claim": "TRUE", "parse_ok": True}, "quality": {"naturalness": 4}}]
    result = summarize(rows)[0]
    assert result["false_claim_acceptance_asr"] == 1.0
    assert result["naturalness_1to5"] == 4.0
