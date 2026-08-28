from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from cta.question_bench import file_sha256
from cta.scei_attack import registered_evidence_text
from cta.scei_graphsearch import (
    build_candidate_bank,
    build_scene_profile,
    graphsearch_scei_events,
    select_next_decision,
)


def _history(decision, feedback: str) -> dict:
    return {
        "stage": "attack",
        "decision": decision.to_dict(),
        "feedback_class": feedback,
    }


def test_scene_profile_prefers_object_compatible_constraints() -> None:
    vehicle = build_scene_profile("airplane", ["airplane", "person"])
    bowl = build_scene_profile("bowl", ["bowl", "spoon"])
    assert vehicle.family_scores[0][0] == "causal_order"
    assert "motion" in vehicle.affordances
    assert bowl.family_scores[0][0] == "capacity_conservation"
    assert "capacity" in bowl.affordances


def test_candidate_bank_is_frozen_deterministic_and_mechanically_valid() -> None:
    args = dict(
        target_label="airplane",
        visible_labels=["airplane", "person"],
        source_sha256="a" * 64,
        seed=20260828,
        max_families=3,
    )
    first_profile, first, first_hash = build_candidate_bank(**args)
    second_profile, second, second_hash = build_candidate_bank(**args)
    assert first_profile == second_profile
    assert first_hash == second_hash
    assert [arm.to_dict() for arm in first] == [arm.to_dict() for arm in second]
    assert [arm.difficulty for arm in first[:2]] == ["moderate", "strong"]
    assert all(abs(arm.record.false_residual) > arm.record.tolerance for arm in first)
    assert all(abs(arm.record.true_residual) <= arm.record.tolerance for arm in first)


def test_policy_separates_delivery_failure_from_semantic_resistance() -> None:
    _, arms, _ = build_candidate_bank(
        "airplane", ["airplane"], source_sha256="b" * 64, max_families=3
    )
    initial = select_next_decision(arms, [])
    assert initial.semantic_index == 0 and initial.delivery_index == 0
    unread = select_next_decision(arms, [_history(initial, "not_read_or_partial")])
    assert unread.semantic_index == initial.semantic_index
    assert unread.delivery_index == 1
    resisted = select_next_decision(arms, [_history(initial, "read_but_resisted")])
    assert arms[resisted.semantic_index].family == arms[initial.semantic_index].family
    assert arms[resisted.semantic_index].difficulty == "strong"
    assert resisted.delivery_index == initial.delivery_index


def test_clean_failure_skips_entire_constraint_family() -> None:
    _, arms, _ = build_candidate_bank(
        "airplane", ["airplane"], source_sha256="c" * 64, max_families=3
    )
    initial = select_next_decision(arms, [])
    failed = {
        "stage": "clean_gate",
        "decision": initial.to_dict(),
        "feedback_class": "clean_gate_failed",
    }
    following = select_next_decision(arms, [failed])
    assert arms[following.semantic_index].family != arms[initial.semantic_index].family
    assert arms[following.semantic_index].difficulty == "moderate"
    assert following.requires_clean_gate is True

    failed_again = {
        "stage": "clean_gate",
        "decision": following.to_dict(),
        "feedback_class": "clean_gate_failed",
    }
    third = select_next_decision(arms, [failed, failed_again])
    assert arms[third.semantic_index].family not in {
        arms[initial.semantic_index].family,
        arms[following.semantic_index].family,
    }


class _GraphVictim:
    def __init__(self, read_outputs: list[str]) -> None:
        self.answer_calls = 0
        self.read_calls = 0
        self.read_outputs = read_outputs

    def infer(self, image_path: str, prompt: str) -> str:
        if prompt.startswith("Transcribe"):
            value = self.read_outputs[self.read_calls]
            self.read_calls += 1
            return value
        self.answer_calls += 1
        # clean(moderate), attacked(moderate), clean(strong), attacked(strong)
        return "YES" if self.answer_calls == 4 else "NO"


def test_graphsearch_reruns_clean_gate_when_semantic_arm_changes(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    Image.new("RGB", (960, 640), (120, 145, 170)).save(source)
    _, bank, bank_hash = build_candidate_bank(
        "airplane", ["airplane"], source_sha256=file_sha256(source), max_families=3
    )
    victim = _GraphVictim([
        registered_evidence_text(bank[0].record, "false"),
        registered_evidence_text(bank[1].record, "false"),
    ])
    events = list(graphsearch_scei_events(
        source,
        "airplane",
        victim,
        tmp_path / "run",
        visible_labels=["airplane"],
        max_rounds=4,
        max_families=3,
    ))
    assert [row["stage"] for row in events] == ["clean_gate", "attack", "clean_gate", "attack"]
    attacks = [row for row in events if row["stage"] == "attack"]
    assert attacks[0]["feedback_class"] == "read_but_resisted"
    assert attacks[0]["arm"]["difficulty"] == "moderate"
    assert attacks[1]["arm"]["difficulty"] == "strong"
    assert attacks[1]["success"] is True
    protocol = json.loads((tmp_path / "run" / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert protocol["candidate_bank_frozen_before_victim"] is True
    assert protocol["candidate_bank_sha256"] == bank_hash
    assert summary["semantic_arms_clean_gated"] == 2
    assert summary["victim_query_count"] == 6
    assert summary["success"] is True
