#!/usr/bin/env python3
"""Create deterministic clean/SoM/render examples for the local SceneTAP chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def panel(path: str, size: tuple[int, int]) -> Image.Image:
    return ImageOps.pad(
        Image.open(path).convert("RGB"), size, method=Image.Resampling.LANCZOS,
        color="white", centering=(0.5, 0.5),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    plans_path = args.plans.resolve()
    renders_path = args.renders.resolve()
    plans = {str(row["question_id"]): row for row in read_jsonl(plans_path)}
    renders = {str(row["question_id"]): row for row in read_jsonl(renders_path)}
    shared = sorted(set(plans) & set(renders))[: args.limit]
    if len(shared) != args.limit:
        raise ValueError(f"requested {args.limit} examples, found {len(shared)}")
    width, height = 320, 220
    header, footer, gap = 34, 46, 10
    canvas = Image.new(
        "RGB", (3 * width + 4 * gap, len(shared) * (height + footer) + header + (len(shared) + 1) * gap),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for col, title in enumerate(("Clean image", "Numbered SoM", "TextDiffuser render")):
        draw.text((gap + col * (width + gap) + 8, 10), title, fill="black", font=font)
    metadata = []
    for row_index, question_id in enumerate(shared):
        plan = plans[question_id]
        render = renders[question_id]
        top = header + gap + row_index * (height + footer + gap)
        paths = (
            plan["original_image_path"], plan["segmentation_image_path"], render["image_path"],
        )
        for col, path in enumerate(paths):
            canvas.paste(panel(path, (width, height)), (gap + col * (width + gap), top))
        footer_text = (
            f"qid={question_id}; fixed text={render['adversarial_text']}; "
            f"region={plan['plan']['text_position_number']}; candidate=0"
        )
        draw.text((gap + 8, top + height + 11), footer_text, fill="#222222", font=font)
        metadata.append({
            "question_id": question_id,
            "selection_rule": "ascending question_id before victim outcomes",
            "adversarial_text": render["adversarial_text"],
            "region": plan["plan"]["text_position_number"],
            "region_resolution": plan.get("region_resolution"),
            "caption_resolution": plan.get("caption_resolution"),
            "selected_candidate_index": render["selected_candidate_index"],
            "source_paths": list(paths),
            "source_sha256": [sha256(Path(path)) for path in paths],
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / "scenetap_local_qwen_examples.png"
    canvas.save(image_path, format="PNG")
    evidence = {
        "schema_version": "cta/scenetap-local-qwen-examples-v1",
        "selection_rule": "first two shared question IDs in ascending lexical order; no victim outputs used",
        "plans_sha256": sha256(plans_path),
        "renders_sha256": sha256(renders_path),
        "image": str(image_path),
        "image_sha256": sha256(image_path),
        "examples": metadata,
    }
    (args.output_dir / "scenetap_local_qwen_examples.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "examples": len(shared)}, indent=2))


if __name__ == "__main__":
    main()
