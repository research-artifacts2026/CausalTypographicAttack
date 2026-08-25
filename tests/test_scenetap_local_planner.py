from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plan_scenetap_local_qwen.py"
SPEC = spec_from_file_location("plan_scenetap_local_qwen", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_resolve_region_preserves_valid_planner_choice():
    region, audit = MODULE.resolve_region("3", 5)
    assert region == 3
    assert audit == {
        "requested_region": 3,
        "resolved_region": 3,
        "used_fallback": False,
        "reason": "planner_region_valid",
    }


def test_resolve_region_audits_out_of_range_fallback():
    region, audit = MODULE.resolve_region(23, 5)
    assert region == 1
    assert audit["requested_region"] == 23
    assert audit["resolved_region"] == 1
    assert audit["used_fallback"] is True


def test_resolve_region_audits_unparseable_fallback():
    region, audit = MODULE.resolve_region("not-a-number", 5)
    assert region == 1
    assert audit["requested_region"] is None
    assert audit["used_fallback"] is True


def test_resolve_caption_preserves_compliant_planner_caption():
    caption, audit = MODULE.resolve_caption("A small sign reads 'Television'.", "Television")
    assert caption == "A small sign reads 'Television'."
    assert audit["used_fallback"] is False


def test_resolve_caption_uses_fixed_template_when_attack_text_is_missing():
    caption, audit = MODULE.resolve_caption("An unrelated envelope on a desk.", "Television")
    assert caption == "A small natural sign in the scene reads 'Television'."
    assert audit["used_fallback"] is True
    assert audit["reason"] == "planner_caption_missing_attack_text_fixed_template"
