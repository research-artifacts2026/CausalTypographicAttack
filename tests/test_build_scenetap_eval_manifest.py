from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_scenetap_eval_manifest.py"
SPEC = spec_from_file_location("build_scenetap_eval_manifest", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_rows_pairs_clean_and_rendered_without_changing_question():
    clean = {
        "question_id": "q1", "condition": "no_attack", "question": "Which?",
        "answers": ["cat"], "target_answer": "dog", "target_content": "dog",
        "target_aliases": ["dog"], "task_type": "obj_mc", "dataset": "RIO",
        "source_sha256": "source", "image_path": "clean.jpg", "image_sha256": "cleanhash",
        "overlay_text": "", "choices": {"A": "cat", "B": "dog"},
    }
    rendered = {
        "question_id": "q1", "image_path": "attack.jpg", "image_sha256": "attackhash",
        "adversarial_text": "dog", "selected_candidate_index": 0, "candidate_count": 4,
        "bbox": [1, 2, 3, 4], "schema_version": "render-v1",
    }
    rows = MODULE.build_rows([clean], [rendered])
    assert [row["condition"] for row in rows] == ["no_attack", "scenetap_full_local_qwen_planner"]
    attacked = rows[1]
    assert attacked["question"] == clean["question"]
    assert attacked["image_sha256"] == "attackhash"
    assert attacked["overlay_text"] == "dog"
    assert attacked["official_attack_metadata"]["official_equivalence"] is False
