import importlib.util
import json
from pathlib import Path

from PIL import Image

from cta.scei_attack import _carrier_dimensions


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_render_seed_is_truth_specific_and_stable():
    module = load_script("render_contraledger_scenetap.py")
    assert module.stable_seed("item", "false", 7) == module.stable_seed("item", "false", 7)
    assert module.stable_seed("item", "false", 7) != module.stable_seed("item", "true", 7)


def test_strip_render_fields_preserves_registered_semantics():
    module = load_script("build_contraledger_flat_baseline.py")
    row = {
        "image_path": "old.jpg",
        "image_sha256": "abc",
        "mask_path": "old.png",
        "mask_sha256": "def",
        "carrier_quad": [[0, 0]],
        "overlay_area_fraction": 0.1,
        "renderer": "old",
        "question": "q",
        "registered_read_text": "fields",
        "correct_semantic": "inconsistent",
    }
    stripped = module.strip_render_fields(row)
    assert set(stripped) == {"question", "registered_read_text", "correct_semantic"}


def test_stage_slug_is_portable():
    module = load_script("stage_contraledger_scenetap.py")
    assert module.safe_slug("voc/item:1") == "voc-item-1"


def test_exact_mcnemar_is_symmetric():
    module = load_script("analyze_contraledger_delivery_matrix.py")
    assert module.exact_mcnemar_p(0, 0) == 1.0
    assert module.exact_mcnemar_p(3, 8) == module.exact_mcnemar_p(8, 3)
    assert 0.0 <= module.exact_mcnemar_p(0, 10) <= 1.0


def test_carrier_dimensions_fit_extreme_aspect_ratios():
    for size in ((333, 1280), (1280, 213), (768, 768)):
        image = Image.new("RGB", size)
        width, height = _carrier_dimensions(image, 0.15)
        assert width > 0 and height > 0
        assert width <= image.width - 20
        assert height <= image.height - 20
        assert width * height <= int(0.15 * image.width * image.height)


def test_source_disjointness_is_fail_closed():
    module = load_script("analyze_contraledger_delivery_matrix.py")
    manifests = {
        "A": {"native": [{"condition": "source_absent", "source_sha256": "a"}]},
        "B": {"native": [{"condition": "source_absent", "source_sha256": "b"}]},
    }
    assert module.assert_source_disjoint(manifests) == {"A/B": 0}
    manifests["B"]["native"][0]["source_sha256"] = "a"
    try:
        module.assert_source_disjoint(manifests)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("source overlap was not rejected")
