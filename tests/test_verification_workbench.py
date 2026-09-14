from pathlib import Path
import json
import shutil
import pytest
from PIL import Image

from cta.verification_demo import demo_rows
from cta.verification_workbench import (CONDITIONS, GOLD, STRATEGIES, archive_session, evaluate_packet,
    freeze_packet, load_packet, parse_option, read_jsonl, score, select_items)
from cta.scei_attack import REQUESTED_COUNTERFACTUAL_FAMILIES
from cta.contraledger_threeway import option_map


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (640, 480), "#98a9a4").save(path)
    return path


@pytest.fixture
def packet(source, tmp_path):
    rows = demo_rows(source, "oven", "unit_conversion", tmp_path / "render")
    return freeze_packet(rows, tmp_path / "packet", origin="unit test")


@pytest.mark.parametrize("bad", ["A or B", "A record is visible. Answer: B", "The answer is B", "", None, "ABC", "A. B.", "(A", "B)"])
def test_parser_rejects_ambiguous_or_incidental_letters(bad):
    assert parse_option(bad, option_map(0)) is None


def test_parser_permutations():
    for i in range(6):
        for letter, semantic in option_map(i).items():
            assert parse_option(f"({letter}).", option_map(i)) == semantic


@pytest.mark.parametrize("family", REQUESTED_COUNTERFACTUAL_FAMILIES)
def test_upload_all_families_fit_and_validate(source, tmp_path, family):
    rows = demo_rows(source, "oven", family, tmp_path / family)
    path = freeze_packet(rows, tmp_path / (family + "-packet"), origin="test")
    info, loaded = load_packet(path)
    assert info["items"] == 1
    assert len({Image.open(r["image_path"]).size for r in loaded}) == 1
    assert loaded[1]["mask_sha256"] == loaded[2]["mask_sha256"]


def test_packet_portable_and_detects_asset_edit(packet, tmp_path):
    moved = tmp_path / "moved"
    shutil.copytree(packet, moved)
    _, rows = load_packet(moved)
    path = Path(rows[0]["image_path"])
    assert path.is_relative_to(moved)
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="asset changed"):
        load_packet(moved)


def test_empty_and_duplicate_manifest_rejected(tmp_path, packet):
    with pytest.raises(ValueError, match="empty"):
        freeze_packet([], tmp_path / "empty", origin="test")
    _, rows = load_packet(packet)
    with pytest.raises(ValueError, match="duplicate"):
        freeze_packet(rows + [rows[0]], tmp_path / "dupe", origin="test")


def test_metrics_zero_eligibility_and_full_population():
    rows = [{"item_id": i} for i in ("one", "two")]
    predictions = []
    for i in ("one", "two"):
        for c in CONDITIONS:
            predictions.append({"item_id": i, "condition": c, "strategy": "direct",
                                "parsed": "consistent"})
    result = score(rows, predictions, ["direct"])["strategies"]["direct"]
    assert result["dc_asr"]["rate"] is None
    assert result["dc_asr"]["n"] == 0
    assert result["accuracy"]["record_false"]["n"] == 2
    assert result["balanced_valid_false_accuracy"] == .5
    assert result["pair_accuracy"]["k"] == 0
    with pytest.raises(ValueError, match="prediction cells"):
        score(rows, predictions[:-1], ["direct"])


class FakeModel:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def infer(self, image, prompt, max_new_tokens):
        self.calls += 1
        if self.fail:
            raise RuntimeError("integration test failure")
        return "A"

    def provenance(self):
        return {"adapter": "test-only"}


def test_run_journal_resume_drift_and_call_accounting(packet, tmp_path):
    model = FakeModel()
    run = tmp_path / "run"
    summary = evaluate_packet(packet, run, {"model": "test"}, lambda cfg: model)
    assert model.calls == 14
    assert summary["actual_calls"] == 14
    assert len(read_jsonl(run / "predictions.jsonl")) == 9
    evaluate_packet(packet, run, {"model": "test"}, lambda cfg: pytest.fail("should not reload on completed replay"))
    assert model.calls == 14
    with pytest.raises(ValueError, match="resume refused"):
        evaluate_packet(packet, run, {"model": "changed"}, lambda cfg: model)
    assert not (run / ".running").exists()
    archive = archive_session(packet, run)
    assert archive.is_file()
    assert json.loads((run / "identity.json").read_text())["parser"] == "bare-option-v1"


def test_failed_calls_retained_and_not_retried(packet, tmp_path):
    model = FakeModel(fail=True)
    run = tmp_path / "failed"
    result = evaluate_packet(packet, run, {}, lambda cfg: model)
    assert result["status"] == "complete_with_errors"
    assert result["failed_calls"] == 14
    assert result["strategies"]["direct"]["unparsed_decisions"] == 3
    assert result["strategies"]["direct"]["dc_asr"]["rate"] is None
    evaluate_packet(packet, run, {}, lambda cfg: pytest.fail("errors must not silently retry"))


def test_family_selection_independent_of_input_order():
    rows = [{"item_id": f"{f}{i}", "family": f, "condition": c} for f in "ab" for i in range(4) for c in CONDITIONS]
    selected = select_items(rows, 2)
    assert {r["item_id"] for r in selected} == {r["item_id"] for r in select_items(rows[::-1], 2)}
    assert {r["family"] for r in selected} == {"a", "b"}


def test_unknown_interrupted_call_is_not_silently_repeated(packet, tmp_path):
    run = tmp_path / "interrupted"
    run.mkdir()
    (run / "pending_call.json").write_text('{}')
    with pytest.raises(ValueError, match="unknown outcome"):
        evaluate_packet(packet, run, {}, lambda cfg: pytest.fail("must not load"))
