from cta.generation import AttackTextGenerator, extract_json
from cta.data import load_dataset
from cta.metrics import claim_matches_overlay, label_match, parse_task_output, summarize
from PIL import Image
import json
import urllib.error
from pathlib import Path

from cta.strong_attack import (
    candidate_policies,
    claim_for_policy,
    render_strong_attack,
    split_sample_ids,
    split_samples_stratified,
)
from cta.model import OpenAIResponsesAdapter
from cta.rvta_bench import (
    render_area_matched_direct_control,
    render_benign_true_evidence,
    validate_annotation_record,
)
from cta.question_bench import (
    CONDITIONS,
    answer_score,
    build_spec,
    choose_target,
    normalize_answer,
    parse_typod_options,
    render_condition,
    scenetap_compatible_score,
    summarize_question_rows,
    target_matches_any,
)
from cta.rio_bench import (
    prediction_letter, rio_mc_score, stable_reservoir, target_letter_from_attack_word,
)
from cta.simulated_capture import PROFILES, simulate_capture
from cta.violation_catalog import SEVERITY_LEVELS, claims_for_label


def test_attack_semantics():
    texts = AttackTextGenerator(42).generate("car", "abc")
    assert "car" in texts["causal"].text.lower()
    assert texts["causal"].violation_type
    assert texts["naive"].target_wrong_label != "car"


def test_requested_violation_severity_examples_are_predeclared():
    person = [claim for level in SEVERITY_LEVELS for claim in claims_for_label("person", level)]
    apple = [claim for level in SEVERITY_LEVELS for claim in claims_for_label("apple", level)]
    assert any("fly forever" in claim.text for claim in person)
    assert any("70 C" in claim.text for claim in person)
    price = next(claim for claim in apple if "USD 3,000" in claim.text)
    assert price.family == "economic/common-sense"
    assert price.physical_impossibility is False
    assert any("70 C" in claim.text for claim in apple)


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


def test_unknown_dataset_rejected():
    try:
        load_dataset("unknown", ".", 1, 1)
    except ValueError as exc:
        assert "Unsupported dataset" in str(exc)
    else:
        raise AssertionError("unknown dataset was accepted")


def test_strong_attack_policy_space_and_claims():
    policies = candidate_policies()
    assert len(policies) == 24
    assert len({policy.policy_id for policy in policies}) == len(policies)
    claim, auxiliary, family = claim_for_policy("car", policies[0])
    assert "car" in claim.lower()
    assert auxiliary
    assert family == "energy/transport"


def test_registered_split_is_deterministic_and_disjoint():
    ids = [f"sample-{index:03d}" for index in range(30)]
    first = split_sample_ids(ids, seed=7, discovery_n=8, test_n=12)
    second = split_sample_ids(list(reversed(ids)), seed=7, discovery_n=8, test_n=12)
    assert first == second
    assert not (set(first["discovery"]) & set(first["test"]))


def test_registered_ablation_split_is_disjoint():
    samples = [
        {"sample_id": f"car-{index:03d}", "target_label": "car"}
        for index in range(40)
    ] + [
        {"sample_id": f"dog-{index:03d}", "target_label": "dog"}
        for index in range(40)
    ]
    split = split_samples_stratified(samples, seed=19, discovery_n=10, test_n=20, ablation_n=30)
    assert len(split["ablation"]) == 30
    assert not (set(split["discovery"]) & set(split["test"]))
    assert not (set(split["discovery"]) & set(split["ablation"]))
    assert not (set(split["test"]) & set(split["ablation"]))


def test_registered_budgeted_test_split_is_disjoint():
    samples = [
        {"sample_id": f"car-{index:03d}", "target_label": "car"}
        for index in range(60)
    ] + [
        {"sample_id": f"dog-{index:03d}", "target_label": "dog"}
        for index in range(60)
    ]
    split = split_samples_stratified(
        samples,
        seed=23,
        discovery_n=10,
        test_n=20,
        ablation_n=30,
        budgeted_test_n=40,
    )
    assert len(split["budgeted_test"]) == 40
    names = ("discovery", "test", "ablation", "budgeted_test")
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            assert not (set(split[left]) & set(split[right]))


def test_stratified_split_covers_available_families():
    samples = [
        {"sample_id": "car-1", "target_label": "car"},
        {"sample_id": "car-2", "target_label": "bus"},
        {"sample_id": "dog-1", "target_label": "dog"},
        {"sample_id": "dog-2", "target_label": "cat"},
        {"sample_id": "pizza-1", "target_label": "pizza"},
        {"sample_id": "pizza-2", "target_label": "cake"},
        {"sample_id": "tv-1", "target_label": "tv"},
        {"sample_id": "tv-2", "target_label": "laptop"},
        {"sample_id": "chair-1", "target_label": "chair"},
        {"sample_id": "chair-2", "target_label": "couch"},
    ]
    split = split_samples_stratified(samples, seed=11, discovery_n=5, test_n=5)
    assert len(split["discovery"]) == 5
    assert len({value.split("-")[0] for value in split["discovery"]}) == 5
    assert not (set(split["discovery"]) & set(split["test"]))


def test_strong_renderer_records_geometry_and_hash(tmp_path):
    source = tmp_path / "source.jpg"
    output = tmp_path / "attacked.jpg"
    Image.new("RGB", (640, 480), (120, 150, 180)).save(source)
    rendered = render_strong_attack(str(source), "car", candidate_policies()[0], output)
    assert output.exists()
    assert rendered.bbox[2] > rendered.bbox[0]
    assert 0 < rendered.overlay_area_fraction < 1
    assert len(rendered.rendered_sha256) == 64


def test_rvta_controls_match_reference_geometry_and_truth(tmp_path):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (640, 480), (120, 150, 180)).save(source)
    reference = {
        "bbox": [10, 12, 390, 190],
        "policy_id": "v2-telemetry-plaque-compact",
        "placement": "lowest-variance-top-left",
    }
    direct = render_area_matched_direct_control(
        str(source), "car", reference, tmp_path / "direct.jpg",
    )
    benign = render_benign_true_evidence(
        str(source), "car", reference, tmp_path / "benign.jpg",
    )
    assert direct.bbox == benign.bbox == tuple(reference["bbox"])
    assert direct.overlay_area_fraction == benign.overlay_area_fraction
    assert direct.expected_claim == "FALSE"
    assert benign.expected_claim == "TRUE"
    assert len(direct.rendered_sha256) == len(benign.rendered_sha256) == 64


def test_rvta_controls_accept_legacy_attack_identifier(tmp_path):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (768, 512), "white").save(source)
    reference = {
        "attack": "v2-telemetry-plaque-compact",
        "bbox": [10, 10, 379, 176],
        "placement": "lowest-variance-top-left",
    }
    direct = render_area_matched_direct_control(
        str(source), "car", reference, tmp_path / "direct.jpg",
    )
    benign = render_benign_true_evidence(
        str(source), "car", reference, tmp_path / "benign.jpg",
    )
    assert direct.control_reference_policy == reference["attack"]
    assert benign.control_reference_policy == reference["attack"]


def test_summary_separates_false_asr_from_true_utility():
    false_row = {
        "attack": "false-card", "defense": "none", "object_correct": True,
        "parsed": {"claim": "TRUE", "parse_ok": True}, "quality": {},
        "expected_claim": "FALSE", "attack_success": True,
        "true_claim_acceptance": False, "claim_correct": False,
        "claim_matches_overlay": True,
    }
    true_row = {
        "attack": "true-card", "defense": "none", "object_correct": True,
        "parsed": {"claim": "TRUE", "parse_ok": True}, "quality": {},
        "expected_claim": "TRUE", "attack_success": False,
        "true_claim_acceptance": True, "claim_correct": True,
        "claim_matches_overlay": True,
    }
    result = {row["attack"]: row for row in summarize([false_row, true_row])}
    assert result["false-card"]["false_claim_acceptance_asr"] == 1.0
    assert result["false-card"]["true_claim_acceptance"] is None
    assert result["true-card"]["false_claim_acceptance_asr"] is None
    assert result["true-card"]["true_claim_acceptance"] == 1.0


def test_rvta_annotation_schema_rejects_ambiguous_values():
    valid = {
        "item_id": "opaque-1", "sample_id": "sample-1", "annotator_id": "ann-1",
        "referent_grounded": True, "visual_relation": "compatible",
        "world_status": "impossible", "naturalness_1to5": 4,
        "scene_fit_1to5": 4, "impossibility_1to5": 5,
        "ambiguity_reason": None,
    }
    validate_annotation_record(valid)
    invalid = dict(valid, world_status="unlikely")
    try:
        validate_annotation_record(invalid)
    except ValueError as exc:
        assert "world_status" in str(exc)
    else:
        raise AssertionError("invalid world status was accepted")


def test_openai_adapter_uses_data_url_and_enforces_query_budget(tmp_path, monkeypatch):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (32, 24), (20, 40, 60)).save(source)
    monkeypatch.setenv("TEST_OPENAI_KEY", "test-only-not-a-real-secret")
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({
                "id": "resp_test",
                "model": "gpt-5.6-sol",
                "status": "completed",
                "output": [{"type": "message", "content": [{
                    "type": "output_text", "text": '{"object":"car","claim_text":"x","claim":"FALSE"}',
                }]}],
                "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
            }).encode()

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    adapter = OpenAIResponsesAdapter({
        "adapter": "openai_responses",
        "model": "gpt-5.6-sol",
        "api_key_env": "TEST_OPENAI_KEY",
        "max_queries": 1,
        "reasoning_effort": "medium",
    })
    output = adapter.infer(str(source), "Return JSON")
    assert '"claim":"FALSE"' in output
    assert captured["payload"]["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,")
    assert captured["payload"]["store"] is False
    assert adapter.inference_metadata()["returned_model"] == "gpt-5.6-sol"
    assert "test-only-not-a-real-secret" not in json.dumps(adapter.provenance())
    try:
        adapter.infer(str(source), "Return JSON")
    except RuntimeError as exc:
        assert "budget exhausted" in str(exc)
    else:
        raise AssertionError("query budget was not enforced")


def test_openai_adapter_redacts_error_message(tmp_path, monkeypatch):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (16, 16), "white").save(source)
    monkeypatch.setenv("TEST_OPENAI_KEY", "do-not-log-this-value")
    body = json.dumps({"error": {
        "message": "Incorrect API key provided: do-not-log-this-value",
        "type": "invalid_request_error",
        "code": "invalid_api_key",
    }}).encode()

    class FakeHttpError(urllib.error.HTTPError):
        def read(self, *_):
            return body

    def fake_with_body(*_args, **_kwargs):
        raise FakeHttpError("https://api.openai.com/v1/responses", 401, "Unauthorized", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fake_with_body)
    adapter = OpenAIResponsesAdapter({
        "api_key_env": "TEST_OPENAI_KEY", "max_queries": 1, "max_retries": 0,
    })
    try:
        adapter.infer(str(source), "Return JSON")
    except RuntimeError as exc:
        rendered = str(exc)
        assert "invalid_api_key" in rendered
        assert "do-not-log-this-value" not in rendered
        assert "Incorrect API key" not in rendered
    else:
        raise AssertionError("HTTP error was not raised")


def test_question_target_is_deterministic_and_not_correct():
    record = {
        "question_id": "q-1", "text": "What animal is shown?",
        "answer": "dog", "choices": ["dog", "cat", "horse"],
    }
    first = choose_target(record, ("dog",), "object", 19)
    second = choose_target(record, ("dog",), "object", 19)
    assert first == second
    assert normalize_answer(first[0]) != "dog"
    assert first[1] == "provided-distractor"


def test_question_builder_renders_complete_paired_conditions(tmp_path):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (640, 480), (120, 150, 180)).save(source)
    record = {
        "question_id": "q-2", "image": source.name,
        "text": "What color is the car?", "answer": "red",
        "choices": ["red", "blue"], "category": "color",
    }
    spec = build_spec(record, tmp_path, seed=7)
    assert spec.task_type == "color"
    assert spec.target_answer == "blue"
    assert "red" in spec.causal_claim and "blue" in spec.causal_claim
    rendered = [
        render_condition(spec, condition, tmp_path / condition / "q-2.jpg")
        for condition in CONDITIONS
    ]
    assert all(Path(item["image_path"]).is_file() for item in rendered)
    assert rendered[0]["bbox"] is None
    assert all(item["bbox"] is not None for item in rendered[1:])
    assert rendered[3]["bbox"] == rendered[4]["bbox"]
    assert spec.causal_claim.lower().count(spec.target_answer.lower()) == 1


def test_question_score_and_clean_conditioned_asr():
    assert answer_score("The dog", ["dog"]) == 1.0
    rows = []
    for qid, clean_correct in (("a", True), ("b", False)):
        rows.append({
            "question_id": qid, "condition": "no_attack",
            "answer_score": float(clean_correct), "target_match": False,
        })
        for condition in CONDITIONS[1:]:
            rows.append({
                "question_id": qid, "condition": condition,
                "answer_score": 0.0, "target_match": condition == "evidence_cta",
            })
    summary = {row["condition"]: row for row in summarize_question_rows(rows)}
    assert summary["evidence_cta"]["n_clean_correct"] == 1
    assert summary["evidence_cta"]["clean_conditioned_asr"] == 1.0
    assert summary["evidence_cta"]["targeted_asr"] == 1.0


def test_typod_binary_options_drive_content_and_public_scoring(tmp_path):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (640, 480), "white").save(source)
    question = "What animal is shown? (a) dog (b) cat"
    assert parse_typod_options(question) == {"a": "dog", "b": "cat"}
    spec = build_spec({
        "question_id": "typod-1", "image": source.name, "text": question,
        "answer": "a", "category": "object recognition",
    }, tmp_path, seed=9)
    assert spec.correct_content == "dog"
    assert spec.target_answer == "b"
    assert spec.target_content == "cat"
    assert "dog" in spec.causal_claim and "cat" in spec.causal_claim
    assert scenetap_compatible_score("(a) dog", ("a",), question, "typo_base_complex") == 1.0
    assert scenetap_compatible_score("cat", ("a",), question, "typo_base_complex") == 0.0
    assert target_matches_any("The answer is (b), cat.", spec.target_aliases)


def test_rio_mc_target_and_scoring_use_attack_word(tmp_path):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (640, 480), "white").save(source)
    row = {
        "question_id": 19,
        "image": source.name,
        "question": "Which object is present? (A) Dog (B) Cat (C) Bus (D) Car. Answer with only the option letter.",
        "answer": "D",
        "choices": {"A": "Dog", "B": "Cat", "C": "Bus", "D": "Car"},
        "attack_word": "Bus",
        "task_type": "object",
    }
    target, source_name = target_letter_from_attack_word(row, seed=3)
    assert target == "c"
    assert source_name == "rio-attack-word-option"
    spec = build_spec({**row, "text": row["question"], "target_answer": target}, tmp_path, seed=3)
    assert spec.correct_content == "Car"
    assert spec.target_content == "Bus"
    assert spec.target_answer == "c"
    assert rio_mc_score("The answer is (D).", "D", row["choices"]) == 1.0
    assert rio_mc_score("Bus", "D", row["choices"]) == 0.0
    assert prediction_letter("I choose C, bus.", row["choices"]) == "c"


def test_stable_rio_selection_is_order_invariant():
    rows = [{"question_id": value} for value in range(20)]
    first = [row["question_id"] for row in stable_reservoir(rows, 5, 11)]
    second = [row["question_id"] for row in stable_reservoir(reversed(rows), 5, 11)]
    assert first == second


def test_simulated_capture_is_deterministic_and_labeled(tmp_path):
    source = tmp_path / "source.jpg"
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    Image.new("RGB", (640, 480), (130, 170, 190)).save(source)
    metadata = simulate_capture(source, first, PROFILES["medium"], seed=17)
    simulate_capture(source, second, PROFILES["medium"], seed=17)
    assert first.read_bytes() == second.read_bytes()
    assert metadata["profile"]["name"] == "medium"
    assert metadata["downscaled_size"][0] < metadata["original_size"][0]
