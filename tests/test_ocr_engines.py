from __future__ import annotations

import pytest

from cta.ocr_engines import normalize_easyocr_detections


def test_normalize_easyocr_detections_filters_by_score() -> None:
    rows = normalize_easyocr_detections(
        [
            ([[1, 2], [9, 2], [9, 7], [1, 7]], "visible", 0.91),
            ([[2, 3], [8, 3], [8, 6], [2, 6]], "weak", 0.49),
        ],
        0.5,
    )
    assert rows == [{
        "box": [[1.0, 2.0], [9.0, 2.0], [9.0, 7.0], [1.0, 7.0]],
        "text": "visible",
        "score": 0.91,
    }]


def test_normalize_easyocr_detections_rejects_malformed_box() -> None:
    with pytest.raises(ValueError, match="invalid box"):
        normalize_easyocr_detections([([1, 2], "bad", 0.8)], 0.5)
