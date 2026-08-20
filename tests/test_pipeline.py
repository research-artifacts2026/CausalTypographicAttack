from cta.generation import AttackTextGenerator, extract_json
from cta.data import load_dataset
from cta.metrics import claim_matches_overlay, label_match, parse_task_output, summarize
from PIL import Image
import json
import urllib.error

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
