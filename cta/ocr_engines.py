from __future__ import annotations

from typing import Iterable


def normalize_easyocr_detections(
    results: Iterable[object], score_threshold: float,
) -> list[dict]:
    """Normalize ``easyocr.Reader.readtext(detail=1)`` output.

    Malformed entries are rejected instead of silently producing a mask.  The
    returned schema matches the detector records used by the existing
    RapidOCR defense implementation.
    """

    detections: list[dict] = []
    for index, item in enumerate(results):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(f"EasyOCR result {index} is not a (box, text, score) triple")
        box, text, score = item
        score = float(score)
        if score < score_threshold:
            continue
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            raise ValueError(f"EasyOCR result {index} has an invalid box")
        points = []
        for point in box:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"EasyOCR result {index} has an invalid point")
            points.append([float(point[0]), float(point[1])])
        detections.append({"box": points, "text": str(text), "score": score})
    return detections
