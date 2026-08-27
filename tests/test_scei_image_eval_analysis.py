from scripts.analyze_scei_image_eval import (
    CONDITIONS,
    analyze_model,
    exact_mcnemar,
    public_analysis,
    public_prediction_row,
)


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


def test_public_analysis_removes_private_paths() -> None:
    result = analyze_model(_rows(), expected_items=3)
    result.update({
        "predictions": "/private/server/predictions.jsonl",
        "predictions_sha256": "a" * 64,
        "provenance": "/private/server/provenance.json",
        "provenance_sha256": "b" * 64,
        "model_provenance": {
            "adapter": "qwen25vl",
            "model_path": "/private/model/cache",
            "device": "cuda:7",
            "generation": {"do_sample": False},
        },
    })
    release = public_analysis({
        "status": "complete",
        "expected_items": 3,
        "models": {"Qwen": result},
        "manifest_sha256": "c" * 64,
        "primary_population": "clean-correct",
        "strict_success": "yes and exact read",
        "claim_boundary": "digital only",
    })
    encoded = str(release)
    assert "/private/" not in encoded
    assert "cuda:7" not in encoded
    assert release["models"]["Qwen"]["adapter"] == "qwen25vl"


def test_public_prediction_row_keeps_outputs_but_drops_paths() -> None:
    row = _rows()[0] | {
        "answer_raw": "NO",
        "read_raw": "",
        "image_sha256": "a" * 64,
        "image_path": "/private/image.jpg",
        "source_path": "/private/source.jpg",
        "mask_path": "/private/mask.png",
        "source_dataset_manifest": "/private/manifest.jsonl",
        "inference_metadata": {"model_path": "/private/model"},
    }
    public = public_prediction_row(row)
    assert public["answer_raw"] == "NO"
    assert public["image_sha256"] == "a" * 64
    assert "image_path" not in public
    assert "source_path" not in public
    assert "mask_path" not in public
    assert "source_dataset_manifest" not in public
    assert "inference_metadata" not in public
