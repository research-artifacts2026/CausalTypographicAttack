#!/usr/bin/env python3
"""Generate cross-model/dataset and natural-render paper assets from raw logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ATTACKS = ["naive", "scene_coherent", "causal"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_raw_run(path: Path, expected_samples: int) -> list[dict]:
    rows = [row for row in read_jsonl(path) if row["defense"] == "none"]
    ids = {row["sample_id"] for row in rows}
    cells = {(row["attack"], row["sample_id"]) for row in rows}
    expected = {(attack, sample_id) for attack in ["none", *ATTACKS] for sample_id in ids}
    if len(ids) != expected_samples or cells != expected or len(rows) != 4 * expected_samples:
        raise ValueError(f"incomplete raw run: {path} ({len(ids)} ids, {len(rows)} rows)")
    return rows


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for attack in ["none", *ATTACKS]:
        cell = [row for row in rows if row["attack"] == attack]
        result[attack] = {
            "n": len(cell),
            "object_accuracy": sum(bool(row["object_correct"]) for row in cell) / len(cell),
            "strict_asr": None if attack == "none" else sum(bool(row["attack_success"]) for row in cell) / len(cell),
            "grounded_transcription": None if attack == "none" else sum(bool(row["claim_matches_overlay"]) for row in cell) / len(cell),
        }
    return result


def validate_single_attack(path: Path, attack: str, expected_samples: int) -> list[dict]:
    rows = [row for row in read_jsonl(path) if row["attack"] == attack and row["defense"] == "none"]
    if len(rows) != expected_samples or len({row["sample_id"] for row in rows}) != expected_samples:
        raise ValueError(f"incomplete {attack} run: {path}")
    return rows


def bootstrap_paired_difference(left: dict[str, bool], right: dict[str, bool], seed: int) -> dict:
    ids = sorted(set(left) & set(right))
    if len(ids) != len(left) or len(ids) != len(right):
        raise ValueError("natural-render runs do not contain identical sample IDs")
    rng = random.Random(seed)
    samples = []
    for _ in range(10000):
        draw = [rng.choice(ids) for _ in ids]
        samples.append(sum(float(right[i]) - float(left[i]) for i in draw) / len(draw))
    samples.sort()
    return {
        "mean": sum(float(right[i]) - float(left[i]) for i in ids) / len(ids),
        "ci95": [samples[249], samples[9749]], "resamples": 10000, "seed": seed,
    }


def tex_percent(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.1f}"


def write_cross_table(path: Path, runs: list[dict]) -> None:
    lines = [
        "% AUTO-GENERATED from completed raw JSONL logs; do not edit",
        "\\begin{tabular}{llrrrrr}",
        "Dataset & Model & Clean Acc. & Naive ASR & Plaque ASR & CTA read & CTA ASR \\\\",
        "\\hline",
    ]
    for run in runs:
        values = run["metrics"]
        lines.append(
            f"{run['dataset']} & {run['model']} & {tex_percent(values['none']['object_accuracy'])} & "
            f"{tex_percent(values['naive']['strict_asr'])} & {tex_percent(values['scene_coherent']['strict_asr'])} & "
            f"{tex_percent(values['causal']['grounded_transcription'])} & {tex_percent(values['causal']['strict_asr'])} \\\\"
        )
    lines += ["\\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_renderer_table(path: Path, pil: dict, natural: dict, paired: dict) -> None:
    lines = [
        "% AUTO-GENERATED from completed matched JSONL logs; do not edit",
        "\\begin{tabular}{lrrr}",
        "Renderer & $N$ & Grounded text & Strict CTA ASR \\\\",
        "\\hline",
        f"PIL plaque & {pil['n']} & {tex_percent(pil['grounded'])} & {tex_percent(pil['asr'])} \\\\ ",
        f"SceneTAP TextDiffuser & {natural['n']} & {tex_percent(natural['grounded'])} & {tex_percent(natural['asr'])} \\\\ ",
        "\\hline",
        f"Paired difference & {pil['n']} & -- & {100 * paired['mean']:+.1f} [{100 * paired['ci95'][0]:+.1f}, {100 * paired['ci95'][1]:+.1f}] \\\\ ",
        "\\end{tabular}", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def make_renderer_grid(path: Path, pil_rows: list[dict], natural_rows: list[dict]) -> dict:
    pil = {row["sample_id"]: row for row in pil_rows}
    natural = {row["sample_id"]: row for row in natural_rows}
    ids = sorted(pil)
    categories = [
        ("TextDiffuser-only success", lambda i: natural[i]["attack_success"] and not pil[i]["attack_success"]),
        ("Both succeed", lambda i: natural[i]["attack_success"] and pil[i]["attack_success"]),
        ("TextDiffuser failure", lambda i: not natural[i]["attack_success"]),
    ]
    selected = []
    used = set()
    for label, predicate in categories:
        sample_id = next((i for i in ids if i not in used and predicate(i)), None)
        if sample_id is None:
            continue
        selected.append((label, sample_id)); used.add(sample_id)
    if len(selected) < 2:
        raise ValueError("insufficient distinct renderer outcome categories for a qualitative grid")
    cell_w, image_h, header_h, result_h, claim_h = 360, 220, 54, 48, 62
    canvas = Image.new(
        "RGB",
        (len(selected) * cell_w, header_h + 2 * (image_h + result_h) + claim_h),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font, body_font, small_font = font(18, True), font(15), font(13)
    for column, (category, sample_id) in enumerate(selected):
        x0 = column * cell_w
        draw.text((x0 + 10, 7), category, fill="black", font=title_font)
        draw.text((x0 + 10, 31), sample_id, fill=(75, 75, 75), font=small_font)
        for row_index, (name, record) in enumerate((("PIL plaque", pil[sample_id]), ("TextDiffuser component", natural[sample_id]))):
            y0 = header_h + row_index * (image_h + result_h)
            image = Image.open(record["image_path"]).convert("RGB")
            image.thumbnail((cell_w - 12, image_h - 8), Image.Resampling.LANCZOS)
            x = x0 + (cell_w - image.width) // 2
            y = y0 + (image_h - image.height) // 2
            canvas.paste(image, (x, y))
            outcome = "success" if record["attack_success"] else "failure"
            draw.text((x0 + 10, y0 + image_h + 3), f"{name}: {outcome}", fill="black", font=body_font)
            draw.text((x0 + 10, y0 + image_h + 25), f"read={record['claim_matches_overlay']}, judge={record['parsed']['claim']}", fill="black", font=small_font)
        claim = pil[sample_id]["attack_text"]
        claim_y = header_h + 2 * (image_h + result_h) + 4
        draw.text((x0 + 10, claim_y), claim[:54], fill=(55, 55, 55), font=small_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return {"selected": [{"category": label, "sample_id": sample_id} for label, sample_id in selected]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-coco", type=Path, required=True)
    parser.add_argument("--qwen7-coco", type=Path, required=True)
    parser.add_argument("--llava-coco", type=Path, required=True)
    parser.add_argument("--intern-coco", type=Path, required=True)
    parser.add_argument("--qwen-voc", type=Path, required=True)
    parser.add_argument("--qwen7-voc", type=Path, required=True)
    parser.add_argument("--llava-voc", type=Path, required=True)
    parser.add_argument("--compact-pil", type=Path, required=True)
    parser.add_argument("--textdiffuser", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    specifications = [
        ("COCO", "Qwen2.5-VL-3B", args.qwen_coco),
        ("COCO", "Qwen2.5-VL-7B", args.qwen7_coco),
        ("COCO", "LLaVA-OV-1.5-8B", args.llava_coco),
        ("COCO", "InternVL2-8B", args.intern_coco),
        ("VOC 2012", "Qwen2.5-VL-3B", args.qwen_voc),
        ("VOC 2012", "Qwen2.5-VL-7B", args.qwen7_voc),
        ("VOC 2012", "LLaVA-OV-1.5-8B", args.llava_voc),
    ]
    runs = []
    for dataset, model, path in specifications:
        runs.append({"dataset": dataset, "model": model, "source": str(path),
                     "source_sha256": sha256_file(path),
                     "metrics": aggregate(validate_raw_run(path, 300))})
    pil_rows = validate_single_attack(args.compact_pil, "causal_compact", 100)
    natural_rows = validate_single_attack(args.textdiffuser, "causal_compact_textdiffuser", 100)
    pil_by_id = {row["sample_id"]: bool(row["attack_success"]) for row in pil_rows}
    natural_by_id = {row["sample_id"]: bool(row["attack_success"]) for row in natural_rows}
    renderer = {}
    for name, rows in (("PIL plaque", pil_rows), ("SceneTAP TextDiffuser", natural_rows)):
        renderer[name] = {"n": len(rows), "asr": sum(bool(row["attack_success"]) for row in rows) / len(rows),
                          "grounded": sum(bool(row["claim_matches_overlay"]) for row in rows) / len(rows)}
    paired = bootstrap_paired_difference(pil_by_id, natural_by_id, 20260312)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    write_cross_table(output / "generated_cross_dataset_table.tex", runs)
    write_renderer_table(output / "generated_renderer_table.tex", renderer["PIL plaque"], renderer["SceneTAP TextDiffuser"], paired)
    grid = make_renderer_grid(output / "figures" / "natural_renderer_examples.png", pil_rows, natural_rows)
    evidence = {"schema_version": "cta/extended-evidence-v1", "model_dataset": runs,
                "renderer": renderer, "renderer_paired_textdiffuser_minus_pil": paired,
                "renderer_sources": {
                    "compact_pil": {"path": str(args.compact_pil), "sha256": sha256_file(args.compact_pil)},
                    "textdiffuser": {"path": str(args.textdiffuser), "sha256": sha256_file(args.textdiffuser)},
                },
                "renderer_grid": grid}
    (output / "extended_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
