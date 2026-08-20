#!/usr/bin/env python3
"""Validate frozen-policy test logs and generate paper-ready evidence assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.strong_attack import BASELINE_POLICY_ID


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_ci(baseline: dict[str, bool], selected: dict[str, bool], seed: int, draws: int = 10000) -> dict:
    ids = sorted(set(baseline) & set(selected))
    if not ids:
        raise ValueError("no matched identifiers for paired interval")
    delta = {sample_id: float(selected[sample_id]) - float(baseline[sample_id]) for sample_id in ids}
    rng = random.Random(seed)
    samples = [statistics.fmean(delta[rng.choice(ids)] for _ in ids) for _ in range(draws)]
    return {
        "n": len(ids),
        "mean": statistics.fmean(delta.values()),
        "ci95": [percentile(samples, 0.025), percentile(samples, 0.975)],
        "draws": draws,
        "seed": seed,
    }


def font(size: int, bold: bool = False):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def fit(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(copy, ((width - copy.width) // 2, (height - copy.height) // 2))
    return canvas


def make_examples(
    output: Path,
    manifest_rows: list[dict],
    selected_policy: str,
    logs: dict[str, list[dict]],
) -> list[dict]:
    images = {row["sample_id"]: row for row in manifest_rows if row["attack"] == selected_policy}
    outcomes: dict[str, dict[str, dict[str, bool]]] = defaultdict(dict)
    for model, rows in logs.items():
        by_key = {(row["sample_id"], row["attack"]): bool(row["attack_success"]) for row in rows}
        for sample_id in images:
            outcomes[sample_id][model] = {
                "baseline": by_key[(sample_id, BASELINE_POLICY_ID)],
                "selected": by_key[(sample_id, selected_policy)],
            }
    model_count = len(logs)
    ranked = []
    for sample_id in sorted(images):
        selected_successes = sum(outcomes[sample_id][model]["selected"] for model in logs)
        improvements = sum(
            outcomes[sample_id][model]["selected"] and not outcomes[sample_id][model]["baseline"]
            for model in logs
        )
        divergence = int(0 < selected_successes < model_count)
        ranked.append((sample_id, improvements, divergence, selected_successes))
    chosen: list[str] = []
    criteria = [
        lambda row: (row[1], row[3]),
        lambda row: (row[2], row[3]),
        lambda row: (-row[3], -row[1]),
    ]
    for criterion in criteria:
        for row in sorted(ranked, key=criterion, reverse=True):
            if row[0] not in chosen:
                chosen.append(row[0])
                break
    card_width, image_height, card_height = 430, 300, 430
    canvas = Image.new("RGB", (card_width * len(chosen), card_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, text_font = font(19, True), font(13)
    records = []
    for index, sample_id in enumerate(chosen):
        row = images[sample_id]
        image = fit(Image.open(row["image_path"]).convert("RGB"), card_width - 20, image_height)
        x0 = index * card_width
        canvas.paste(image, (x0 + 10, 8))
        y = image_height + 16
        draw.text((x0 + 12, y), sample_id, fill="#111111", font=title_font)
        y += 26
        for model, result in outcomes[sample_id].items():
            line = f"{model}: old={'T' if result['baseline'] else 'F'}  v2={'T' if result['selected'] else 'F'}"
            draw.text((x0 + 12, y), line, fill="#b00020" if result["selected"] else "#176b32", font=text_font)
            y += 19
        for line in textwrap.wrap(row["attack_text"], 48)[:2]:
            draw.text((x0 + 12, y), line, fill="#333333", font=text_font)
            y += 18
        records.append({"sample_id": sample_id, "outcomes": outcomes[sample_id], "image_path": row["image_path"]})
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--model-log", action="append", required=True, help="LABEL=predictions.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    test_manifest = args.test_manifest.resolve()
    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if split_manifest["active_split"] != "test":
        raise ValueError("split manifest is not a test split")
    if set(split_manifest["discovery_ids"]) & set(split_manifest["test_ids"]):
        raise ValueError("discovery/test identifier leakage")
    selection_path = args.selection.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_policy = selection["selected_policy_id"]
    manifest_rows = read_jsonl(test_manifest)
    expected = {(row["sample_id"], row["attack"], row["defense"]) for row in manifest_rows}
    expected_attacks = {BASELINE_POLICY_ID, selected_policy}
    if {row["attack"] for row in manifest_rows} != expected_attacks:
        raise ValueError("test manifest must contain exactly baseline and frozen selected policy")

    logs: dict[str, list[dict]] = {}
    log_sources = []
    for spec in args.model_log:
        label, raw_path = spec.split("=", 1)
        path = Path(raw_path).resolve()
        rows = read_jsonl(path)
        observed = {(row["sample_id"], row["attack"], row["defense"]) for row in rows}
        if observed != expected:
            raise ValueError(f"incomplete model log {label}: missing={len(expected-observed)} extra={len(observed-expected)}")
        provenance_path = path.with_name("provenance.json")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("completed_rows") != len(expected) or not provenance.get("finished_at_utc"):
            raise ValueError(f"unfinished provenance for {label}")
        logs[label] = rows
        log_sources.append({
            "model": label,
            "path": str(path),
            "sha256": sha256(path),
            "provenance": str(provenance_path),
            "provenance_sha256": sha256(provenance_path),
        })

    evidence = {
        "schema_version": "cta/strong-test-evidence-v1",
        "selected_policy_id": selected_policy,
        "test_samples": len(set(split_manifest["active_ids"])),
        "test_manifest": str(test_manifest),
        "test_manifest_sha256": sha256(test_manifest),
        "selection": str(selection_path),
        "selection_sha256": sha256(selection_path),
        "models": {},
        "sources": log_sources,
    }
    for model_index, (model, rows) in enumerate(logs.items()):
        by_attack = defaultdict(list)
        for row in rows:
            by_attack[row["attack"]].append(row)
        model_result = {"attacks": {}}
        for attack, attack_rows in sorted(by_attack.items()):
            model_result["attacks"][attack] = {
                "n": len(attack_rows),
                "grounded": statistics.fmean(bool(row["claim_matches_overlay"]) for row in attack_rows),
                "strict_asr": statistics.fmean(bool(row["attack_success"]) for row in attack_rows),
                "object_accuracy": statistics.fmean(bool(row["object_correct"]) for row in attack_rows),
                "mean_overlay_area_fraction": statistics.fmean(
                    float(row["attack_metadata"].get("overlay_area_fraction", 0.0)) for row in attack_rows
                ),
            }
        baseline = {row["sample_id"]: bool(row["attack_success"]) for row in by_attack[BASELINE_POLICY_ID]}
        selected = {row["sample_id"]: bool(row["attack_success"]) for row in by_attack[selected_policy]}
        model_result["paired_selected_minus_baseline"] = paired_ci(baseline, selected, args.seed + model_index)
        evidence["models"][model] = model_result

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    table = [
        "% AUTO-GENERATED from complete frozen-policy JSONL logs; do not edit",
        "\\begin{tabular}{lrrrr}",
        "Model & Old CTA & Evidence CTA & $\\Delta$ [95\\% CI] & Grounded \\\\",
        "\\hline",
    ]
    for model, result in evidence["models"].items():
        old = 100 * result["attacks"][BASELINE_POLICY_ID]["strict_asr"]
        new = 100 * result["attacks"][selected_policy]["strict_asr"]
        paired = result["paired_selected_minus_baseline"]
        grounded = 100 * result["attacks"][selected_policy]["grounded"]
        table.append(
            f"{model} & {old:.1f} & {new:.1f} & {100*paired['mean']:+.1f} "
            f"[{100*paired['ci95'][0]:+.1f}, {100*paired['ci95'][1]:+.1f}] & {grounded:.1f} \\\\"
        )
    table += ["\\end{tabular}", ""]
    (output_dir / "generated_strong_test_table.tex").write_text("\n".join(table), encoding="utf-8")
    evidence["qualitative_examples"] = make_examples(
        output_dir / "figures" / "strong_test_examples.png", manifest_rows, selected_policy, logs,
    )
    (output_dir / "strong_test_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_policy_id": selected_policy,
        "models": list(evidence["models"]),
        "output_dir": str(output_dir),
    }))


if __name__ == "__main__":
    main()
