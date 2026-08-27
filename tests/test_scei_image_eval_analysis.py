from scripts.analyze_scei_image_eval import CONDITIONS, analyze_model, exact_mcnemar


def _rows() -> list[dict]:
    result = []
    attack_answers = {
        "a": ("yes", "yes"),
        "b": ("no", "yes"),
        "c": ("yes", "no"),
    }
    for item_id, (flat_answer, scene_answer) in attack_answers.items():
        for condition in CONDITIONS:
            if condition == "clean_false":
                parsed = "no"
            elif condition in {"clean_true", "scene_true"}:
                parsed = "yes"
            elif condition == "flat_false":
                parsed = flat_answer
            else:
                parsed = scene_answer
            result.append({
                "item_id": item_id,
                "condition": condition,
                "parsed_semantic": parsed,
                "exact_read_match": condition in {"flat_false", "scene_false", "scene_true"},
                "family": "range_threshold",
                "counterbalance_cell": "false:yesno:semantic",
            })
    return result


def test_paired_scene_flat_analysis() -> None:
    result = analyze_model(_rows(), expected_items=3)
    paired = result["paired_scene_minus_flat"]
    assert paired["flat_successes"] == 2
    assert paired["scene_successes"] == 2
    assert paired["flat_only"] == 1
    assert paired["scene_only"] == 1
    assert paired["exact_mcnemar_p_two_sided"] == 1.0


def test_exact_mcnemar_handles_no_discordance() -> None:
    assert exact_mcnemar(0, 0) == 1.0
