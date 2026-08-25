#!/usr/bin/env python3
"""Create a deterministic six-condition RVTA-QA qualitative grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


CONDITIONS = (
    "no_attack", "benign_true", "direct_answer",
    "causal_claim", "evidence_cta", "causal_bridge",
)
LABELS = {
    "no_attack": "Clean",
    "benign_true": "Benign true",
    "direct_answer": "Direct answer",
    "causal_claim": "Causal claim",
    "evidence_cta": "Evidence CTA",
    "causal_bridge": "Causal-Bridge",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ("DejaVuSans-Bold.ttf", "Arial Bold.ttf") if bold else ("DejaVuSans.ttf", "Arial.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def short_model(name: str) -> str:
    return (
        name.replace("Qwen2.5-VL-3B", "Q3")
        .replace("Qwen2.5-VL-7B", "Q7")
        .replace("LLaVA-OneVision-1.5-8B", "LV")
        .replace("InternVL2-8B", "IV")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-log", action="append", required=True, help="MODEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest.resolve())
    manifest_by_key = {(row["item_id"], row["condition"]): row for row in manifest}
    if len(manifest_by_key) != len(manifest) or {row["condition"] for row in manifest} != set(CONDITIONS):
        raise ValueError("manifest is duplicate or has the wrong condition set")
    models: dict[str, dict[tuple[str, str], dict]] = {}
    log_metadata = {}
    for assignment in args.model_log:
        model, value = assignment.split("=", 1)
        path = Path(value).resolve()
        rows = read_jsonl(path)
        by_key = {(row["item_id"], row["condition"]): row for row in rows}
        if len(rows) != len(manifest_by_key) or set(by_key) != set(manifest_by_key):
            raise ValueError(f"{model}: incomplete or duplicate prediction log")
        provenance = json.loads((path.parent / "provenance.json").read_text(encoding="utf-8"))
        if provenance.get("status") != "complete" or provenance.get("completed_rows") != len(manifest_by_key):
            raise ValueError(f"{model}: provenance is incomplete")
        for key, row in by_key.items():
            if row["image_sha256"] != manifest_by_key[key]["image_sha256"]:
                raise ValueError(f"{model}: image hash mismatch at {key}")
        models[model] = by_key
        log_metadata[model] = {"path": str(path), "sha256": sha256(path)}

    item_ids = sorted({row["item_id"] for row in manifest})
    candidates = []
    for item_id in item_ids:
        clean_all = all(models[model][(item_id, "no_attack")]["parsed_answer"] == "a" for model in models)
        bridge_read_all = all(bool(models[model][(item_id, "causal_bridge")]["read_match"]) for model in models)
        if not clean_all or not bridge_read_all:
            continue
        bridge_only = sum(
            models[model][(item_id, "causal_bridge")]["parsed_answer"] == "b"
            and models[model][(item_id, "causal_claim")]["parsed_answer"] != "b"
            for model in models
        )
        bridge_success = sum(
            models[model][(item_id, "causal_bridge")]["parsed_answer"] == "b" for model in models
        )
        candidates.append((bridge_only, bridge_success, item_id))
    if not candidates:
        raise ValueError("no item satisfies the preregistered clean/read eligibility rule")
    bridge_only, bridge_success, item_id = sorted(candidates, key=lambda row: (-row[0], -row[1], row[2]))[0]

    cell_w, image_h, header_h, footer_h = 500, 385, 42, 70
    cell_h = header_h + image_h + footer_h
    canvas = Image.new("RGB", (cell_w * 3, cell_h * 2), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, badge_font = font(22, True), font(16, False)
    for index, condition in enumerate(CONDITIONS):
        row_index, col_index = divmod(index, 3)
        x0, y0 = col_index * cell_w, row_index * cell_h
        source = Image.open(manifest_by_key[(item_id, condition)]["image_path"]).convert("RGB")
        fitted = ImageOps.contain(source, (cell_w - 12, image_h - 8), method=Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (cell_w, image_h), "white")
        panel.paste(fitted, ((cell_w - fitted.width) // 2, (image_h - fitted.height) // 2))
        canvas.paste(panel, (x0, y0 + header_h))
        draw.text((x0 + 10, y0 + 8), LABELS[condition], font=title_font, fill="#111111")
        badges = []
        for model, rows in models.items():
            prediction = rows[(item_id, condition)]
            answer = prediction["parsed_answer"].upper()
            read = "R" if prediction.get("read_match") else "-"
            badges.append(f"{short_model(model)}:{answer}/{read}")
        draw.text((x0 + 10, y0 + header_h + image_h + 8), "  ".join(badges), font=badge_font, fill="#202020")
        if condition == "causal_bridge":
            border = "#009E73"
        elif condition == "causal_claim":
            border = "#56B4E9"
        else:
            border = "#777777"
        draw.rectangle((x0 + 2, y0 + 2, x0 + cell_w - 3, y0 + cell_h - 3), outline=border, width=3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, dpi=(300, 300))
    metadata = {
        "schema_version": "cta/rvta-qa-teaser-v1",
        "selection_rule": (
            "Require all-model clean A and all-model bridge read match; maximize bridge-only successes, "
            "then bridge successes, then choose the lexicographically first item."
        ),
        "selected_item_id": item_id,
        "bridge_only_successes": bridge_only,
        "bridge_successes": bridge_success,
        "eligible_candidates": len(candidates),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest.resolve()),
        "logs": log_metadata,
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output.resolve()),
        "badge_format": "model:parsed-answer/read-match; R means exact normalized registered-claim match",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
