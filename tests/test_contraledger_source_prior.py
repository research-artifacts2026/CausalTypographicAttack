from cta.contraledger import (
    CUE_LEVELS,
    decision_semantics,
    source_prior_items,
    summarize_prior_adjusted,
    summarize_source_prior,
)


def _manifest():
    rows = []
    for item_id, polarity in (("a", "positive"), ("b", "negative")):
        for cue in ("values_only", "authority", "explicit_conclusion"):
            for truth in ("true", "false"):
                target = ("yes" if polarity == "positive" else "no") if truth == "false" else None
                rows.append({
                    "item_id": item_id,
                    "condition": f"{cue}_{truth}",
                    "family": "unit_conversion",
                    "question_polarity": polarity,
                    "question": f"question-{item_id}",
                    "source_path": f"{item_id}.jpg",
                    "source_sha256": item_id * 64,
                    "option_order": "yes_no",
                    "target_semantic": target,
                    "correct_semantic": "no" if target == "yes" else "yes",
                })
    return rows


def test_source_prior_deduplicates_factorial_items():
    items = source_prior_items(_manifest())
    assert len(items) == 2
    assert items[0]["false_target_semantic"] == "yes"
    assert items[1]["false_target_semantic"] == "no"


def test_source_prior_summary_exposes_question_prior():
    items = source_prior_items(_manifest())
    items[0]["prior_parsed"] = "yes"
    items[1]["prior_parsed"] = "no"
    result = summarize_source_prior(items)
    assert result["false_target_prior_rate"] == 1.0
    assert result["yes_rate"] == 0.5
    assert result["by_question_polarity"]["positive"]["false_target_prior_rate"] == 1.0


def test_prior_adjusted_summary_separates_question_prior_from_attack_induction():
    predictions = []
    priors = []
    for item_id, polarity, prior_answer in (
        ("a", "positive", "no"),
        ("b", "negative", "no"),
    ):
        false_correct, false_target = decision_semantics("false", polarity)
        priors.append({
            "item_id": item_id,
            "prior_parsed": prior_answer,
            "false_target_semantic": false_target,
        })
        for cue in CUE_LEVELS:
            true_correct, _ = decision_semantics("true", polarity)
            predictions.extend([
                {
                    "item_id": item_id,
                    "condition": f"{cue}_true",
                    "correct_semantic": true_correct,
                    "target_semantic": None,
                    "decide_parsed": true_correct,
                },
                {
                    "item_id": item_id,
                    "condition": f"{cue}_false",
                    "correct_semantic": false_correct,
                    "target_semantic": false_target,
                    "decide_parsed": false_target,
                },
            ])
    result = summarize_prior_adjusted(predictions, priors)
    values = next(row for row in result if row["cue_level"] == "values_only")
    assert values["attacked_false_target_rate"] == 1.0
    assert values["source_prior_target_rate"] == 0.5
    assert values["attack_induction_rate_given_prior_non_target"] == 1.0
