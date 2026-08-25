#!/usr/bin/env python3
"""Render frozen local SceneTAP plans with the official TextDiffuser component."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from utils.get_rectangle_by_mask import largest_inscribed_rectangle
from utils.text_diffuser import TextDiffuser
from utils.typo_attack_planner import find_text_region


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--font-path", default="./fonts/arial.ttf")
    args = parser.parse_args()
    plans_path = args.plans.resolve()
    output_root = args.output_root.resolve()
    manifest = output_root / "render_manifest.jsonl"
    if manifest.exists():
        raise FileExistsError(f"refusing to overwrite {manifest}")
    output_root.mkdir(parents=True, exist_ok=True)
    diffuser = TextDiffuser()
    rows = read_jsonl(plans_path)
    for row in rows:
        started = time.time()
        image = Image.open(row["original_image_path"]).convert("RGB")
        masks = np.load(row["mask_path"], allow_pickle=True)
        region_index = int(row["plan"]["text_position_number"]) - 1
        target_mask = masks[region_index]["segmentation"]
        x, y, width, height = largest_inscribed_rectangle(target_mask, True)
        mask_width, mask_height = target_mask.T.shape
        left = x / mask_width * image.width
        top = y / mask_height * image.height
        right = (x + width) / mask_width * image.width
        bottom = (y + height) / mask_height * image.height
        bbox = find_text_region(
            row["adversarial_text"], left, top, right, bottom,
            font_path=args.font_path, font_size=20, aspect_ratio_threshold=0.1,
        )
        positions = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
        result = diffuser.generate(
            positions,
            row["original_image_path"],
            row["adversarial_text"],
            row["plan"]["short_caption"],
            radio="Two Points",
            scale_factor=2,
            regional_diffusion=True,
        )
        images = [candidate.resize(image.size) for candidate in result[0]]
        selected = output_root / "images" / row["image"]
        selected.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(selected, format="JPEG", quality=95)
        candidates_root = output_root / "candidates" / str(row["question_id"])
        candidates_root.mkdir(parents=True, exist_ok=True)
        for index, candidate in enumerate(images):
            candidate.save(candidates_root / f"{index}.jpg", format="JPEG", quality=95)
        rendered = {
            "schema_version": "cta/scenetap-local-qwen-render-v1",
            **row,
            "condition": "scenetap_full_local_qwen_planner",
            "image_path": str(selected),
            "image_sha256": sha256(selected),
            "bbox": list(bbox),
            "candidate_count": len(images),
            "selected_candidate_index": 0,
            "renderer": "official SceneTAP TextDiffuser component",
            "render_latency_s": round(time.time() - started, 4),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rendered, ensure_ascii=False, sort_keys=True) + "\n")
    provenance = {
        "schema_version": "cta/scenetap-local-qwen-rendering-v1",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plans": str(plans_path),
        "plans_sha256": sha256(plans_path),
        "rows": len(rows),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "official_equivalence": False,
        "boundary": "Official SoM and TextDiffuser components with a local Qwen2.5-VL-7B planner; not the official GPT-4o planner checkpoint/service.",
    }
    (output_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
