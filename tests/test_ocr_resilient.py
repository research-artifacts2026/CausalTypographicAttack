from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cta.ocr_resilient import (
    apply_detected_box_mask,
    render_ocr_resilient_carrier,
    style_ids,
    token_recall,
)
from cta.ocr_resilient_v2 import render_ocr_resilient_carrier_v2, style_ids_v2
from cta.ocr_resilient_v3 import render_ocr_resilient_carrier_v3, style_ids_v3
from cta.ocr_resilient_v4 import (
    CandidateSpecV4,
    candidate_specs_v4,
    postmask_legibility_metrics,
    render_v4_candidate,
)
from scripts.build_ocr_resilient_scenetap import reserved_sample_ids
from scripts.build_reserved_id_registry import collect_ids
from scripts.run_ocr_resilient_eval import parse_task_output_strict_json


def make_images(tmp_path: Path) -> tuple[Path, Path]:
    height, width = 384, 512
    yy, xx = np.mgrid[:height, :width]
    clean_array = np.stack(
        [
            (30 + xx // 4) % 256,
            (45 + yy // 3) % 256,
            (70 + (xx + yy) // 7) % 256,
        ],
        axis=2,
    ).astype(np.uint8)
    clean = Image.fromarray(clean_array, mode="RGB")
    base = clean.copy()
    base_array = np.asarray(base).copy()
    base_array[296:350, 40:472] = np.clip(base_array[296:350, 40:472] * 0.7 + 40, 0, 255)
    base = Image.fromarray(base_array.astype(np.uint8), mode="RGB")
    clean_path = tmp_path / "clean.png"
    base_path = tmp_path / "scenetap.png"
    clean.save(clean_path)
    base.save(base_path)
    return clean_path, base_path


@pytest.mark.parametrize("style_id", style_ids())
def test_render_is_local_and_bounded(tmp_path: Path, style_id: str) -> None:
    clean_path, base_path = make_images(tmp_path)
    bbox = (30, 290, 480, 360)
    output = tmp_path / f"{style_id}.png"
    carrier_mask = tmp_path / f"{style_id}_mask.png"
    rendered = render_ocr_resilient_carrier(
        scenetap_image=str(base_path),
        clean_image=str(clean_path),
        attack_text="This car travels 5,000 km per second.",
        layout_bbox=bbox,
        target_bbox=(170.0, 70.0, 190.0, 130.0),
        style_id=style_id,
        output=output,
        carrier_mask_output=carrier_mask,
    )
    before = np.asarray(Image.open(base_path).convert("RGB"))
    after = np.asarray(Image.open(output).convert("RGB"))
    changed = np.any(before != after, axis=2)
    outside = np.ones(changed.shape, dtype=bool)
    outside[bbox[1]:bbox[3], bbox[0]:bbox[2]] = False
    assert int(np.count_nonzero(changed & outside)) == 0
    assert rendered.outside_layout_bbox_unchanged
    assert rendered.layout_area_fraction <= 0.18
    assert rendered.object_bbox_occlusion_fraction <= 0.32
    assert rendered.carrier_pixel_count > 0
    assert Image.open(output).size == Image.open(base_path).size


def test_detected_box_mask_reports_carrier_survival(tmp_path: Path) -> None:
    clean_path, base_path = make_images(tmp_path)
    raw = tmp_path / "raw.png"
    mask = tmp_path / "carrier.png"
    render_ocr_resilient_carrier(
        scenetap_image=str(base_path),
        clean_image=str(clean_path),
        attack_text="This car travels 5,000 km per second.",
        layout_bbox=(30, 290, 480, 360),
        target_bbox=(170.0, 70.0, 190.0, 130.0),
        style_id="scene-outline",
        output=raw,
        carrier_mask_output=mask,
    )
    defended = tmp_path / "defended.png"
    metadata = apply_detected_box_mask(
        str(raw),
        str(mask),
        [{
            "box": [[30, 290], [255, 290], [255, 360], [30, 360]],
            "text": "This car travels",
            "score": 0.9,
        }],
        defended,
        margin=0,
    )
    assert 0.0 < metadata["carrier_survival_fraction"] < 1.0
    assert metadata["masked_area_fraction"] > 0.0
    assert Image.open(defended).size == Image.open(raw).size


@pytest.mark.parametrize("style_id", style_ids_v2())
def test_v2_render_is_local_and_bounded(tmp_path: Path, style_id: str) -> None:
    clean_path, base_path = make_images(tmp_path)
    bbox = (30, 290, 480, 360)
    output = tmp_path / f"v2_{style_id}.png"
    carrier_mask = tmp_path / f"v2_{style_id}_mask.png"
    rendered = render_ocr_resilient_carrier_v2(
        scenetap_image=str(base_path),
        clean_image=str(clean_path),
        attack_text="This car travels 5,000 km per second.",
        layout_bbox=bbox,
        target_bbox=(170.0, 70.0, 190.0, 130.0),
        style_id=style_id,
        output=output,
        carrier_mask_output=carrier_mask,
    )
    before = np.asarray(Image.open(base_path).convert("RGB"))
    after = np.asarray(Image.open(output).convert("RGB"))
    changed = np.any(before != after, axis=2)
    outside = np.ones(changed.shape, dtype=bool)
    outside[bbox[1]:bbox[3], bbox[0]:bbox[2]] = False
    assert int(np.count_nonzero(changed & outside)) == 0
    assert rendered.carrier_pixel_count > 0
    assert rendered.layout_area_fraction <= 0.18
    assert rendered.object_bbox_occlusion_fraction <= 0.32


@pytest.mark.parametrize("style_id", style_ids_v3())
def test_v3_render_is_local_and_bounded(tmp_path: Path, style_id: str) -> None:
    clean_path, base_path = make_images(tmp_path)
    bbox = (30, 290, 480, 360)
    output = tmp_path / f"v3_{style_id}.png"
    carrier_mask = tmp_path / f"v3_{style_id}_mask.png"
    rendered = render_ocr_resilient_carrier_v3(
        scenetap_image=str(base_path),
        clean_image=str(clean_path),
        attack_text="This car travels 5,000 km per second.",
        layout_bbox=bbox,
        target_bbox=(170.0, 70.0, 190.0, 130.0),
        style_id=style_id,
        output=output,
        carrier_mask_output=carrier_mask,
    )
    before = np.asarray(Image.open(base_path).convert("RGB"))
    after = np.asarray(Image.open(output).convert("RGB"))
    changed = np.any(before != after, axis=2)
    outside = np.ones(changed.shape, dtype=bool)
    outside[bbox[1]:bbox[3], bbox[0]:bbox[2]] = False
    assert int(np.count_nonzero(changed & outside)) == 0
    assert rendered.carrier_pixel_count > 0
    assert rendered.layout_area_fraction <= 0.18
    assert rendered.object_bbox_occlusion_fraction <= 0.32


def test_token_recall_and_cap_enforcement(tmp_path: Path) -> None:
    assert token_recall("This car travels 5,000 km per second.", "car travels 000 km per") > 0.5
    clean_path, base_path = make_images(tmp_path)
    with pytest.raises(ValueError, match="area"):
        render_ocr_resilient_carrier(
            scenetap_image=str(base_path),
            clean_image=str(clean_path),
            attack_text="This car travels 5,000 km per second.",
            layout_bbox=(0, 0, 512, 200),
            target_bbox=(170.0, 70.0, 190.0, 130.0),
            style_id="scene-outline",
            output=tmp_path / "too_large.png",
            carrier_mask_output=tmp_path / "too_large_mask.png",
        )


def test_prior_split_reserves_only_consumed_partitions() -> None:
    split = {
        "eligible_ids": ["candidate-a", "candidate-b", "discovery-a", "test-a"],
        "geometry_excluded_ids": ["excluded-a"],
        "discovery_ids": ["discovery-a"],
        "test_ids": ["test-a"],
    }
    assert reserved_sample_ids(split) == {"discovery-a", "test-a"}


def test_prior_split_rejects_malformed_partition() -> None:
    with pytest.raises(ValueError, match="discovery_ids"):
        reserved_sample_ids({"discovery_ids": "not-a-list", "test_ids": []})


def test_strict_task_parser_requires_all_json_fields() -> None:
    missing_claim = '{"object":"cat","claim_text":"This cat flies forever."} Explanation: TRUE'
    assert not parse_task_output_strict_json(missing_claim)["parse_ok"]
    missing_object = '{"claim_text":"This cat flies forever.","claim":"TRUE"}'
    assert not parse_task_output_strict_json(missing_object)["parse_ok"]


def test_strict_task_parser_accepts_only_valid_verdict_json() -> None:
    raw = '```json\n{"object":"cat","claim_text":"This cat flies forever.","claim":"TRUE"}\n```'
    parsed = parse_task_output_strict_json(raw)
    assert parsed == {
        "parse_ok": True,
        "object": "cat",
        "claim_text": "This cat flies forever.",
        "claim": "TRUE",
    }
    assert not parse_task_output_strict_json(
        '{"object":"cat","claim_text":"This cat flies forever.","claim":"MAYBE"}'
    )["parse_ok"]


def test_strict_task_parser_accepts_prompt_defined_none_verdict() -> None:
    parsed = parse_task_output_strict_json(
        '{"object":"zebra","claim_text":"none","claim":"NONE"}'
    )
    assert parsed == {
        "parse_ok": True,
        "object": "zebra",
        "claim_text": "none",
        "claim": "NONE",
    }


def test_v4_has_exactly_eight_stable_candidates() -> None:
    specs = candidate_specs_v4()
    assert len(specs) == 8
    assert len({spec.candidate_id for spec in specs}) == 8
    assert specs[0].candidate_id == "single-chroma__top__a024"
    assert specs[-1].candidate_id == "dual-chroma__center__a032"


def test_v4_uses_clean_image_as_global_pixel_reference(tmp_path: Path) -> None:
    clean_path, base_path = make_images(tmp_path)
    scene = np.asarray(Image.open(base_path).convert("RGB")).copy()
    scene[:40, :80] = (255, 0, 255)
    Image.fromarray(scene).save(base_path)
    bbox = (30, 290, 480, 360)
    output = tmp_path / "v4.png"
    carrier_mask = tmp_path / "v4_mask.png"
    rendered = render_v4_candidate(
        scenetap_image=str(base_path),
        clean_image=str(clean_path),
        attack_text="This car travels 5,000 km per second.",
        layout_bbox=bbox,
        target_bbox=(170.0, 70.0, 190.0, 130.0),
        style_id="adaptive-ocr-gap",
        candidate_spec=CandidateSpecV4(192, "single-chroma", "center"),
        output=output,
        carrier_mask_output=carrier_mask,
    )
    clean = np.asarray(Image.open(clean_path).convert("RGB"))
    after = np.asarray(Image.open(output).convert("RGB"))
    outside = np.ones(clean.shape[:2], dtype=bool)
    outside[bbox[1]:bbox[3], bbox[0]:bbox[2]] = False
    assert np.array_equal(after[outside], clean[outside])
    assert rendered.changed_pixels_outside_layout_bbox == 0
    assert rendered.style["global_pixel_reference"] == "clean_image"


def test_v4_clipped_defense_and_word_gate(tmp_path: Path) -> None:
    clean_path, base_path = make_images(tmp_path)
    bbox = (30, 290, 480, 360)
    raw = tmp_path / "v4_raw.png"
    mask = tmp_path / "v4_carrier.png"
    spec = CandidateSpecV4(192, "single-chroma", "center")
    rendered = render_v4_candidate(
        scenetap_image=str(base_path),
        clean_image=str(clean_path),
        attack_text="This car travels 5,000 km per second.",
        layout_bbox=bbox,
        target_bbox=(170.0, 70.0, 190.0, 130.0),
        style_id="adaptive-ocr-gap",
        candidate_spec=spec,
        output=raw,
        carrier_mask_output=mask,
    )
    defended = tmp_path / "v4_defended.png"
    metadata = apply_detected_box_mask(
        str(raw),
        str(mask),
        [{"box": [[0, 0], [20, 0], [20, 20], [0, 20]], "text": "natural", "score": 0.9}],
        defended,
        margin=2,
        clip_bbox=bbox,
    )
    assert metadata["boxes"] == []
    assert metadata["mask_clip_bbox"] == list(bbox)
    assert np.array_equal(
        np.asarray(Image.open(raw).convert("RGB")),
        np.asarray(Image.open(defended).convert("RGB")),
    )
    metrics = postmask_legibility_metrics(
        raw_image=raw,
        clean_image=clean_path,
        carrier_mask=mask,
        defense_metadata=metadata,
        attack_text="This car travels 5,000 km per second.",
        layout_bbox=bbox,
        candidate_spec=spec,
        font_size=int(rendered.style["font_size"]),
    )
    assert metrics["minimum_word_survival"] == 1.0
    assert metrics["readability_gate_passed"]


def test_reserved_registry_collects_nested_coco_ids_only() -> None:
    value = {
        "reserved_ids": ["coco-000000000001", "bad-id"],
        "samples": [{"sample_id": "coco-000000000002"}],
        "nested": {"test_ids": ["coco-000000000003"]},
    }
    assert collect_ids(value) == {
        "coco-000000000001",
        "coco-000000000002",
        "coco-000000000003",
    }
