from cta.generation import AttackTextGenerator, extract_json
from cta.data import load_dataset
from cta.metrics import claim_matches_overlay, label_match, parse_task_output, summarize
from cta.strong_attack import candidate_policies, claim_for_policy, split_sample_ids, split_samples_stratified


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
