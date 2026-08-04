#!/usr/bin/env python3
"""Build deployable OCR-mask defense inputs with RapidOCR detections."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image, ImageDraw
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.metrics import STOPWORDS


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOPWORDS and len(token) > 1}


def token_recall(expected: str, observed: str) -> float:
    expected_tokens = tokens(expected)
    observed_tokens = tokens(observed)
    if not expected_tokens:
        return 0.0
    return len(expected_tokens & observed_tokens) / len(expected_tokens)


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg_path = args.config.resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    source_path = Path(cfg["source_log"]).resolve()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    conditions_path = output_root / "conditions.jsonl"
    detections_path = output_root / "detections.jsonl"

    from rapidocr import RapidOCR

    engine = RapidOCR()
    attacks = set(cfg.get("attacks", ["naive", "causal"]))
    score_threshold = float(cfg.get("score_threshold", 0.5))
    margin = int(cfg.get("mask_margin_px", 2))
    source_rows = [
        row for row in read_jsonl(source_path)
        if row["attack"] in attacks and row["defense"] == "none"
    ]
    source_rows.sort(key=lambda row: (row["sample_id"], row["attack"]))
    expected_n = int(cfg.get("expected_samples", 0))
    if expected_n and len({row["sample_id"] for row in source_rows}) != expected_n:
        raise ValueError("source log does not contain the expected number of samples")

    completed = set()
    if conditions_path.exists():
        completed = {(row["sample_id"], row["attack"]) for row in read_jsonl(conditions_path)}
    totals = {attack: {"rows": 0, "overlay_detected": 0, "boxes": 0} for attack in sorted(attacks)}

    for source in tqdm(source_rows, desc="RapidOCR masks"):
        key = (source["sample_id"], source["attack"])
        if key in completed:
            continue
        image = Image.open(source["image_path"]).convert("RGB")
        result = engine(source["image_path"])
        raw_boxes = list(result.boxes or [])
        raw_texts = list(result.txts or [])
        raw_scores = list(result.scores or [])
        kept = []
        for box, text, score in zip(raw_boxes, raw_texts, raw_scores):
            if float(score) < score_threshold:
                continue
            points = [[float(point[0]), float(point[1])] for point in box]
            kept.append({"box": points, "text": str(text), "score": float(score)})

        masked = image.copy()
        draw = ImageDraw.Draw(masked)
        masked_area_upper_bound = 0
        for item in kept:
            xs = [point[0] for point in item["box"]]
            ys = [point[1] for point in item["box"]]
            x0 = max(0, int(min(xs)) - margin)
            y0 = max(0, int(min(ys)) - margin)
            x1 = min(image.width - 1, int(max(xs)) + margin)
            y1 = min(image.height - 1, int(max(ys)) + margin)
            draw.rectangle((x0, y0, x1, y1), fill=(127, 127, 127))
            masked_area_upper_bound += max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)

        output_image = output_root / "images" / source["attack"] / f"{source['sample_id']}.jpg"
        output_image.parent.mkdir(parents=True, exist_ok=True)
        masked.save(output_image, quality=95)
        observed = " ".join(item["text"] for item in kept)
        recall = token_recall(source["attack_text"], observed)
        defense_meta = {
            "engine": "RapidOCR",
            "engine_version": importlib.metadata.version("rapidocr"),
            "score_threshold": score_threshold,
            "mask_margin_px": margin,
            "boxes": kept,
            "recognized_text": observed,
            "overlay_token_recall": recall,
            "overlay_detected_at_0.5_recall": recall >= 0.5,
            "masked_area_upper_bound_fraction": min(1.0, masked_area_upper_bound / (image.width * image.height)),
        }
        condition = {
            **{key_: source[key_] for key_ in ["sample_id", "source_sha256", "target_label", "attack", "attack_text", "attack_metadata"]},
            "defense": "rapidocr_mask",
            "defense_metadata": defense_meta,
            "image_path": str(output_image.resolve()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        append_jsonl(conditions_path, condition)
        append_jsonl(detections_path, {"sample_id": source["sample_id"], "attack": source["attack"], **defense_meta})
        totals[source["attack"]]["rows"] += 1
        totals[source["attack"]]["overlay_detected"] += int(recall >= 0.5)
        totals[source["attack"]]["boxes"] += len(kept)

    final_rows = read_jsonl(conditions_path)
    manifest = {
        "schema_version": "cta/rapidocr-mask-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(cfg_path),
        "config_sha256": sha256(cfg_path),
        "source_log": str(source_path),
        "source_log_sha256": sha256(source_path),
        "completed_rows": len(final_rows),
        "score_threshold": score_threshold,
        "mask_margin_px": margin,
        "engine": "RapidOCR",
        "engine_version": importlib.metadata.version("rapidocr"),
    }
    (output_root / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
