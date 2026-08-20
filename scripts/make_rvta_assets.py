#!/usr/bin/env python3
"""Generate RVTA matched-baseline and held-out factorial evidence from complete logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _font(size: int, bold: bool = False):
    suffix = "-Bold" if bold else ""
    for candidate in (
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf",
        f"/usr/share/fonts/truetype/liberation2/"
        f"LiberationSans{'-Bold' if bold else '-Regular'}.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def make_matched_control_figure(output: Path, manifest_rows: list[dict]) -> Path:
    """Render a fixed-ID geometry control figure without outcome-based selection."""
    sample_id = sorted({row["sample_id"] for row in manifest_rows})[0]
    conditions = (
        ("rvta-area-matched-direct-control", "Direct false claim"),
        ("v2-telemetry-plaque-compact", "Evidence-framed false claim"),
        ("rvta-benign-true-evidence", "Benign true claim"),
    )
    indexed = {(row["sample_id"], row["attack"]): row for row in manifest_rows}
    panel_width, label_height = 360, 52
    prepared = []
    for condition, label in conditions:
        row = indexed[(sample_id, condition)]
        image = Image.open(row["image_path"]).convert("RGB")
        panel_height = round(panel_width * image.height / image.width)
        image = image.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
        prepared.append((label, image))
    image_height = max(image.height for _, image in prepared)
    canvas = Image.new("RGB", (panel_width * len(prepared), label_height + image_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(20, bold=True)
    colors = ((74, 85, 104), (153, 74, 42), (39, 112, 82))
    for index, ((label, image), color) in enumerate(zip(prepared, colors)):
        x = index * panel_width
        draw.rectangle((x, 0, x + panel_width, label_height), fill=color)
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        draw.text((x + (panel_width - text_width) / 2, 14), label, fill="white", font=font)
        canvas.paste(image, (x, label_height))
    figure = output / "rvta_matched_controls.png"
    canvas.save(figure)
    metadata = {
        "selection_rule": "lexicographically first frozen matched-test sample identifier",
        "sample_id": sample_id,
        "conditions": [condition for condition, _ in conditions],
        "source_manifest_sha256": sha256(Path(indexed[(sample_id, conditions[0][0])]["image_path"]).parent.parent.parent / "render_manifest.jsonl"),
        "figure_sha256": sha256(figure),
    }
    (output / "rvta_matched_controls_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    return figure


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_mean(values: dict[str, float], seed: int, draws: int = 10000) -> dict:
    ids = sorted(values)
    if not ids:
        raise ValueError("cannot bootstrap an empty mapping")
    rng = random.Random(seed)
    samples = [statistics.fmean(values[rng.choice(ids)] for _ in ids) for _ in range(draws)]
    return {
        "n": len(ids),
        "mean": statistics.fmean(values.values()),
        "ci95": [percentile(samples, 0.025), percentile(samples, 0.975)],
        "draws": draws,
        "seed": seed,
    }


def parse_labeled(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        label, raw = value.split("=", 1)
        if label in result:
            raise ValueError(f"duplicate model label: {label}")
        result[label] = Path(raw).resolve()
    return result


def validate_log(path: Path, expected: set[tuple[str, str, str]]) -> list[dict]:
    rows = read_jsonl(path)
    observed = {(row["sample_id"], row["attack"], row["defense"]) for row in rows}
    if observed != expected:
        raise ValueError(f"incomplete log {path}: missing={len(expected-observed)} extra={len(observed-expected)}")
    provenance_path = path.with_name("provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("completed_rows") != len(expected) or not provenance.get("finished_at_utc"):
        raise ValueError(f"unfinished provenance: {path}")
    return rows


def strict_rate(rows: list[dict]) -> float:
    if not rows:
        raise ValueError("missing rows for strict rate")
    return statistics.fmean(bool(row["attack_success"]) for row in rows)


def grounded_rate(rows: list[dict]) -> float:
    if not rows:
        raise ValueError("missing rows for grounded rate")
    return statistics.fmean(bool(row["claim_matches_overlay"]) for row in rows)


def matched_evidence(manifest: Path, logs: dict[str, Path], seed: int) -> tuple[dict, list[str]]:
    manifest_rows = read_jsonl(manifest)
    expected = {(row["sample_id"], row["attack"], row["defense"]) for row in manifest_rows}
    attacks = sorted({row["attack"] for row in manifest_rows})
    expected_attacks = {
        "none", "naive", "scene_coherent", "v1-original-causal-plaque",
        "v2-telemetry-plaque-compact", "rvta-area-matched-direct-control",
        "rvta-benign-true-evidence",
    }
    if set(attacks) != expected_attacks:
        raise ValueError(f"unexpected matched condition set: {attacks}")
    manifest_by_attack = defaultdict(list)
    for row in manifest_rows:
        manifest_by_attack[row["attack"]].append(row)
    evidence = {
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "samples": len({row["sample_id"] for row in manifest_rows}),
        "conditions": attacks,
        "models": {},
        "sources": [],
    }
    false_baselines = [
        "naive", "scene_coherent", "v1-original-causal-plaque", "rvta-area-matched-direct-control",
    ]
    selected_attack = "v2-telemetry-plaque-compact"
    for model_index, (model, path) in enumerate(logs.items()):
        rows = validate_log(path, expected)
        by_attack = defaultdict(list)
        for row in rows:
            by_attack[row["attack"]].append(row)
        metrics = {}
        for attack in attacks:
            items = by_attack[attack]
            expected_truth = {row.get("expected_claim") for row in items}
            result = {
                "n": len(items),
                "object_accuracy": statistics.fmean(bool(row["object_correct"]) for row in items),
                "grounded": None if attack == "none" else grounded_rate(items),
                "expected_claim": next(iter(expected_truth)) if len(expected_truth) == 1 else None,
                "mean_overlay_area_fraction": statistics.fmean(
                    float(row["attack_metadata"].get("overlay_area_fraction", 0.0)) for row in items
                ),
            }
            if result["expected_claim"] == "FALSE":
                result["strict_asr"] = strict_rate(items)
            elif result["expected_claim"] == "TRUE":
                result["true_acceptance"] = statistics.fmean(
                    bool(row.get("true_claim_acceptance")) for row in items
                )
            metrics[attack] = result
        selected = {row["sample_id"]: float(row["attack_success"]) for row in by_attack[selected_attack]}
        paired = {}
        for baseline_index, baseline in enumerate(false_baselines):
            baseline_values = {row["sample_id"]: float(row["attack_success"]) for row in by_attack[baseline]}
            if set(selected) != set(baseline_values):
                raise ValueError(f"unmatched identifiers for {model}: {baseline}")
            paired[baseline] = bootstrap_mean(
                {sample_id: selected[sample_id] - baseline_values[sample_id] for sample_id in selected},
                seed + model_index * 10 + baseline_index,
            )
        evidence["models"][model] = {"conditions": metrics, "paired_evidence_minus": paired}
        evidence["sources"].append({"model": model, "path": str(path), "sha256": sha256(path)})
    return evidence, false_baselines


def _policy_factor(policy_id: str) -> tuple[str, str, str]:
    parts = policy_id.split("-")
    if len(parts) != 4 or parts[0] != "v2":
        raise ValueError(f"not a registered v2 policy: {policy_id}")
    return parts[1], parts[2], parts[3]


def ablation_evidence(manifest: Path, logs: dict[str, Path], seed: int) -> dict:
    manifest_rows = read_jsonl(manifest)
    policies = sorted({row["attack"] for row in manifest_rows if row["attack"].startswith("v2-")})
    if len(policies) != 24:
        raise ValueError(f"held-out ablation requires all 24 v2 policies, found {len(policies)}")
    expected = {(row["sample_id"], row["attack"], row["defense"]) for row in manifest_rows}
    evidence = {
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "samples": len({row["sample_id"] for row in manifest_rows}),
        "candidate_policies": len(policies),
        "models": {},
        "sources": [],
    }
    factor_levels = {
        "claim": (0, ("direct", "telemetry", "record")),
        "artifact": (1, ("plaque", "sensor", "certificate", "compound")),
        "scale": (2, ("compact", "large")),
    }
    for model_index, (model, path) in enumerate(logs.items()):
        rows = validate_log(path, expected)
        policy_rows = [row for row in rows if row["attack"] in policies]
        cells = {}
        for policy in policies:
            items = [row for row in policy_rows if row["attack"] == policy]
            cells[policy] = {
                "n": len(items),
                "strict_asr": strict_rate(items),
                "grounded": grounded_rate(items),
                "mean_overlay_area_fraction": statistics.fmean(
                    float(row["attack_metadata"]["overlay_area_fraction"]) for row in items
                ),
            }
        marginals = {}
        for factor, (factor_index, levels) in factor_levels.items():
            marginals[factor] = {}
            for level in levels:
                selected_policies = [policy for policy in policies if _policy_factor(policy)[factor_index] == level]
                by_sample = defaultdict(list)
                area = []
                grounded = []
                for row in policy_rows:
                    if row["attack"] in selected_policies:
                        by_sample[row["sample_id"]].append(float(row["attack_success"]))
                        area.append(float(row["attack_metadata"]["overlay_area_fraction"]))
                        grounded.append(float(row["claim_matches_overlay"]))
                per_sample = {sample_id: statistics.fmean(values) for sample_id, values in by_sample.items()}
                estimate = bootstrap_mean(per_sample, seed + model_index * 100 + len(marginals[factor]))
                estimate["policies"] = len(selected_policies)
                estimate["grounded"] = statistics.fmean(grounded)
                estimate["mean_overlay_area_fraction"] = statistics.fmean(area)
                marginals[factor][level] = estimate
        contrasts = {}
        for name, factor, positive, negative in (
            ("telemetry_minus_direct", "claim", "telemetry", "direct"),
            ("compact_minus_large", "scale", "compact", "large"),
        ):
            positive_policies = [policy for policy in policies if _policy_factor(policy)[factor_levels[factor][0]] == positive]
            negative_policies = [policy for policy in policies if _policy_factor(policy)[factor_levels[factor][0]] == negative]
            positive_by_id, negative_by_id = defaultdict(list), defaultdict(list)
            for row in policy_rows:
                if row["attack"] in positive_policies:
                    positive_by_id[row["sample_id"]].append(float(row["attack_success"]))
                if row["attack"] in negative_policies:
                    negative_by_id[row["sample_id"]].append(float(row["attack_success"]))
            if set(positive_by_id) != set(negative_by_id):
                raise ValueError(f"unmatched held-out contrast: {name}")
            contrasts[name] = bootstrap_mean({
                sample_id: statistics.fmean(positive_by_id[sample_id]) - statistics.fmean(negative_by_id[sample_id])
                for sample_id in positive_by_id
            }, seed + model_index * 1000 + len(contrasts))
        evidence["models"][model] = {"cells": cells, "factor_marginals": marginals, "contrasts": contrasts}
        evidence["sources"].append({"model": model, "path": str(path), "sha256": sha256(path)})
    return evidence


def write_tables(output: Path, matched: dict, false_baselines: list[str], ablation: dict | None) -> None:
    selected = "v2-telemetry-plaque-compact"
    labels = {
        "naive": "Naive",
        "scene_coherent": "Scene-aware",
        "v1-original-causal-plaque": "Original CTA",
        "rvta-area-matched-direct-control": "Area-matched direct",
    }
    table = [
        "% AUTO-GENERATED from complete matched RVTA logs; do not edit",
        "\\begin{tabular}{lrrrrrr}",
        "Model & Naive & Scene & Old CTA & Area ctrl. & Evidence CTA & True utility \\\\",
        "\\hline",
    ]
    for model, result in matched["models"].items():
        conditions = result["conditions"]
        values = [100 * conditions[key]["strict_asr"] for key in false_baselines]
        evidence_asr = 100 * conditions[selected]["strict_asr"]
        utility = 100 * conditions["rvta-benign-true-evidence"]["true_acceptance"]
        table.append(
            f"{model} & {values[0]:.1f} & {values[1]:.1f} & {values[2]:.1f} & "
            f"{values[3]:.1f} & {evidence_asr:.1f} & {utility:.1f} \\\\"
        )
    table += ["\\end{tabular}", ""]
    (output / "generated_rvta_matched_table.tex").write_text("\n".join(table), encoding="utf-8")

    gain_table = [
        "% AUTO-GENERATED paired image-level differences; do not edit",
        "\\begin{tabular}{llr}",
        "Model & Evidence CTA minus baseline & $\\Delta$ [95\\% CI] \\\\",
        "\\hline",
    ]
    for model, result in matched["models"].items():
        for baseline in false_baselines:
            paired = result["paired_evidence_minus"][baseline]
            gain_table.append(
                f"{model} & {labels[baseline]} & {100*paired['mean']:+.1f} "
                f"[{100*paired['ci95'][0]:+.1f}, {100*paired['ci95'][1]:+.1f}] \\\\"
            )
    gain_table += ["\\end{tabular}", ""]
    (output / "generated_rvta_paired_gains_table.tex").write_text("\n".join(gain_table), encoding="utf-8")

    if ablation:
        ablation_table = [
            "% AUTO-GENERATED from complete held-out factorial logs; do not edit",
            "\\begin{tabular}{lllrr}",
            "Model & Factor & Level & ASR & Area \\\\",
            "\\hline",
        ]
        for model, result in ablation["models"].items():
            first = True
            for factor in ("claim", "artifact", "scale"):
                for level, estimate in result["factor_marginals"][factor].items():
                    ablation_table.append(
                        f"{model if first else ''} & {factor.title()} & {level.title()} & "
                        f"{100*estimate['mean']:.1f} & {100*estimate['mean_overlay_area_fraction']:.1f} \\\\"
                    )
                    first = False
        ablation_table += ["\\end{tabular}", ""]
        (output / "generated_rvta_ablation_table.tex").write_text(
            "\n".join(ablation_table), encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-manifest", type=Path, required=True)
    parser.add_argument("--matched-model-log", action="append", required=True)
    parser.add_argument("--ablation-manifest", type=Path)
    parser.add_argument("--ablation-model-log", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    matched, false_baselines = matched_evidence(
        args.matched_manifest.resolve(), parse_labeled(args.matched_model_log), args.seed,
    )
    ablation = None
    if args.ablation_manifest or args.ablation_model_log:
        if not args.ablation_manifest or not args.ablation_model_log:
            raise ValueError("ablation manifest and model logs must be supplied together")
        ablation = ablation_evidence(
            args.ablation_manifest.resolve(), parse_labeled(args.ablation_model_log), args.seed + 10000,
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    make_matched_control_figure(output, read_jsonl(args.matched_manifest.resolve()))
    record = {
        "schema_version": "cta/rvta-benchmark-evidence-v1",
        "matched": matched,
        "held_out_factorial": ablation,
    }
    (output / "rvta_benchmark_evidence.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    write_tables(output, matched, false_baselines, ablation)
    print(json.dumps({
        "matched_models": list(matched["models"]),
        "ablation_models": [] if ablation is None else list(ablation["models"]),
        "output": str(output),
    }))


if __name__ == "__main__":
    main()
