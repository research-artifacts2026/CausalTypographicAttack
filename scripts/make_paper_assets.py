#!/usr/bin/env python3
"""Generate evidence-bearing paper figures/tables from completed CTA logs.

This script never runs a model and never accepts aggregate values on the command
line.  All numbers and qualitative labels are read from sample-level JSONL rows.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import textwrap
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SEED = 2026
N_BOOT = 10_000
SAMPLE_ID = "000000000071"
OKABE_ITO = {
    "naive": "#0072B2",
    "scene_coherent": "#E69F00",
    "causal": "#D55E00",
    "defended": "#009E73",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def keyed(rows: list[dict[str, Any]], attack: str, defense: str) -> dict[str, dict[str, Any]]:
    selected = {
        row["sample_id"]: row
        for row in rows
        if row["attack"] == attack and row["defense"] == defense
    }
    if not selected:
        raise ValueError(f"missing cell attack={attack!r}, defense={defense!r}")
    return selected


def binary_vector(cell: dict[str, dict[str, Any]], ids: list[str]) -> np.ndarray:
    return np.asarray([bool(cell[sid]["attack_success"]) for sid in ids], dtype=float)


def percentile_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    n = len(values)
    draws = values[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def paired_difference_ci(
    lhs: np.ndarray, rhs: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    if len(lhs) != len(rhs):
        raise ValueError("paired arrays have different lengths")
    delta = lhs - rhs
    n = len(delta)
    draws = delta[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(delta.mean()), float(lo), float(hi)


def fmt_ci(stats: dict[str, float]) -> str:
    return f"{100*stats['mean']:.2f} [{100*stats['ci_low']:.2f}, {100*stats['ci_high']:.2f}]"


def compute_statistics(rows: list[dict[str, Any]], log_path: Path) -> dict[str, Any]:
    cells = {
        name: keyed(rows, attack, defense)
        for name, (attack, defense) in {
            "naive_raw": ("naive", "none"),
            "scene_raw": ("scene_coherent", "none"),
            "causal_raw": ("causal", "none"),
            "naive_consistency": ("naive", "consistency"),
            "causal_consistency": ("causal", "consistency"),
            "naive_mask": ("naive", "ocr_mask"),
            "causal_mask": ("causal", "ocr_mask"),
        }.items()
    }
    ids = sorted(cells["causal_raw"])
    if len(ids) != 300:
        raise ValueError(f"expected 300 main samples, found {len(ids)}")
    for name, cell in cells.items():
        if sorted(cell) != ids:
            raise ValueError(f"cell {name} does not contain the same 300 sample ids")

    vectors = {name: binary_vector(cell, ids) for name, cell in cells.items()}
    rng = np.random.default_rng(SEED)
    asr: dict[str, dict[str, float]] = {}
    for name, vector in vectors.items():
        mean, lo, hi = percentile_ci(vector, rng)
        asr[name] = {"mean": mean, "ci_low": lo, "ci_high": hi}

    differences = {}
    for name, lhs, rhs in [
        ("causal_minus_naive_raw", "causal_raw", "naive_raw"),
        ("causal_minus_scene_raw", "causal_raw", "scene_raw"),
        ("naive_defense_reduction", "naive_raw", "naive_consistency"),
        ("causal_defense_reduction", "causal_raw", "causal_consistency"),
    ]:
        mean, lo, hi = paired_difference_ci(vectors[lhs], vectors[rhs], rng)
        differences[name] = {"mean": mean, "ci_low": lo, "ci_high": hi}

    return {
        "schema_version": "cta/statistics-v1",
        "source": str(log_path),
        "source_sha256": sha256(log_path),
        "n_samples": len(ids),
        "bootstrap": {
            "resamples": N_BOOT,
            "seed": SEED,
            "confidence_level_percent": 95,
            "interval": "percentile",
        },
        "asr": asr,
        "paired_differences": differences,
    }


def write_statistics_table(stats: dict[str, Any], path: Path) -> None:
    a = stats["asr"]
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Attack & Raw ASR (95\% CI) $\uparrow$ & Consistency ASR $\downarrow$ & Oracle-mask ASR $\downarrow$ \\",
        r"\midrule",
        f"Naive typography & {fmt_ci(a['naive_raw'])} & {fmt_ci(a['naive_consistency'])} & {fmt_ci(a['naive_mask'])} " + r"\\",
        f"Scene-coherent plaque & {fmt_ci(a['scene_raw'])} & -- & -- " + r"\\",
        f"CTA (ours) & \\textbf{{{fmt_ci(a['causal_raw'])}}} & \\textbf{{{fmt_ci(a['causal_consistency'])}}} & {fmt_ci(a['causal_mask'])} " + r"\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_quality_table(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [("naive", "Naive typography"), ("scene_coherent", "Scene-coherent plaque"), ("causal", "CTA (ours)")]
    means = {}
    for attack, _ in labels:
        cell = [r for r in rows if r["attack"] == attack and r["defense"] == "none"]
        if len(cell) != 300:
            raise ValueError(f"quality cell {attack} has n={len(cell)}")
        # One cached judge row used the legacy key ``scene_coherent`` for the
        # scene-compatibility score.  Preserve the value with an explicit,
        # auditable fallback matching the run aggregator.
        scene_scores = [
            r["quality"].get("visual_scene_compatibility", r["quality"].get("scene_coherent"))
            for r in cell
        ]
        if any(value is None for value in scene_scores):
            raise ValueError(f"quality cell {attack} has a missing scene score")
        means[attack] = {
            "visual_scene_compatibility": float(np.mean(scene_scores)),
            "naturalness": float(np.mean([r["quality"]["naturalness"] for r in cell])),
            "reality_violation": float(np.mean([r["quality"]["reality_violation"] for r in cell])),
        }
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Attack & Scene compatibility $\uparrow$ & Naturalness $\uparrow$ & Reality violation $\uparrow$ \\",
        r"\midrule",
    ]
    for attack, label in labels:
        m = means[attack]
        values = [m["visual_scene_compatibility"], m["naturalness"], m["reality_violation"]]
        if attack == "causal":
            lines.append(f"{label} & {values[0]:.2f} & {values[1]:.2f} & \\textbf{{{values[2]:.2f}}} " + r"\\")
        else:
            lines.append(f"{label} & {values[0]:.2f} & {values[1]:.2f} & {values[2]:.2f} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_result_plot(stats: dict[str, Any], output_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    a = stats["asr"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.35), gridspec_kw={"wspace": 0.30})

    names = ["Naive", "Scene-coherent", "CTA"]
    keys = ["naive_raw", "scene_raw", "causal_raw"]
    colors = [OKABE_ITO["naive"], OKABE_ITO["scene_coherent"], OKABE_ITO["causal"]]
    means = np.array([100 * a[k]["mean"] for k in keys])
    errors = np.array(
        [[means[i] - 100 * a[k]["ci_low"] for i, k in enumerate(keys)], [100 * a[k]["ci_high"] - means[i] for i, k in enumerate(keys)]]
    )
    bars = axes[0].bar(names, means, color=colors, yerr=errors, capsize=3, width=0.68)
    axes[0].set_title("(a) Raw attack strength")
    axes[0].set_ylabel("Strict ASR (%, higher is stronger)")
    axes[0].set_ylim(0, 80)
    axes[0].grid(axis="y", alpha=0.2, linewidth=0.5)
    for bar, value in zip(bars, means):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 3.0, f"{value:.1f}", ha="center", va="bottom", fontsize=7)

    x = np.arange(2)
    width = 0.34
    raw_keys = ["naive_raw", "causal_raw"]
    def_keys = ["naive_consistency", "causal_consistency"]
    raw_means = np.array([100 * a[k]["mean"] for k in raw_keys])
    def_means = np.array([100 * a[k]["mean"] for k in def_keys])
    raw_err = np.array(
        [[raw_means[i] - 100 * a[k]["ci_low"] for i, k in enumerate(raw_keys)], [100 * a[k]["ci_high"] - raw_means[i] for i, k in enumerate(raw_keys)]]
    )
    def_err = np.array(
        [[def_means[i] - 100 * a[k]["ci_low"] for i, k in enumerate(def_keys)], [100 * a[k]["ci_high"] - def_means[i] for i, k in enumerate(def_keys)]]
    )
    b1 = axes[1].bar(x - width / 2, raw_means, width, label="No defense", color="#999999", yerr=raw_err, capsize=3)
    b2 = axes[1].bar(x + width / 2, def_means, width, label="Consistency wrapper", color=OKABE_ITO["defended"], yerr=def_err, capsize=3)
    axes[1].set_title("(b) Object-consistency defense")
    axes[1].set_xticks(x, ["Naive", "CTA"])
    axes[1].set_ylabel("Strict ASR (%, lower is safer)")
    axes[1].set_ylim(0, 80)
    axes[1].grid(axis="y", alpha=0.2, linewidth=0.5)
    axes[1].legend(loc="upper left", frameon=False)
    for bars_ in [b1, b2]:
        for bar in bars_:
            value = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2, value + 3.0, f"{value:.1f}", ha="center", va="bottom", fontsize=7)

    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=350, bbox_inches="tight")
    plt.close(fig)


def fit_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def make_qualitative_grid(pilot_rows: list[dict[str, Any]], image_root: Path, output: Path) -> None:
    selected = {
        row["attack"]: row
        for row in pilot_rows
        if row["sample_id"] == SAMPLE_ID and row["defense"] == "none"
    }
    if set(selected) != {"none", "naive", "scene_coherent", "causal"}:
        raise ValueError("qualitative sample is missing one or more raw conditions")
    if not selected["causal"]["attack_success"] or selected["naive"]["attack_success"] or selected["scene_coherent"]["attack_success"]:
        raise ValueError("qualitative sample no longer has the intended matched-case outcome")
    if selected["none"]["parsed"]["object"].lower() != selected["none"]["target_label"].lower():
        raise ValueError("qualitative clean prediction is not correct")

    panel_w, panel_h = 760, 575
    header_h, footer_h, gap, margin = 60, 75, 24, 24
    canvas = Image.new("RGB", (2 * panel_w + gap + 2 * margin, 2 * panel_h + gap + 2 * margin), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = fit_font(28, bold=True)
    body_font = fit_font(21)
    small_font = fit_font(18)
    panels = [
        ("none", "(a) Clean image", "Visual control"),
        ("naive", "(b) Naive typography", "Wrong-object claim"),
        ("scene_coherent", "(c) Scene-coherent plaque", "Wrong-object claim"),
        ("causal", "(d) Causal typography (ours)", "Correct referent, impossible event"),
    ]
    for idx, (attack, title, description) in enumerate(panels):
        row_idx, col_idx = divmod(idx, 2)
        x0 = margin + col_idx * (panel_w + gap)
        y0 = margin + row_idx * (panel_h + gap)
        border = OKABE_ITO["causal"] if attack == "causal" else "#666666"
        draw.rounded_rectangle((x0, y0, x0 + panel_w, y0 + panel_h), radius=10, fill="#FAFAFA", outline=border, width=4)
        draw.text((x0 + 18, y0 + 12), title, font=title_font, fill="#111111")
        # Keep the header single-purpose at final paper width.  The semantic
        # role is stated in the log-derived footer and the figure caption.

        folder = "none" if attack == "none" else attack
        image_path = image_root / folder / f"{SAMPLE_ID}.jpg"
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((panel_w - 24, panel_h - header_h - footer_h - 12), Image.Resampling.LANCZOS)
        ix = x0 + (panel_w - image.width) // 2
        iy = y0 + header_h + 5
        canvas.paste(image, (ix, iy))

        parsed = selected[attack]["parsed"]
        footer_y = y0 + panel_h - footer_h + 8
        if attack == "none":
            verdict = f"Target: {selected[attack]['target_label']}   |   Predicted object: {parsed['object']}"
            verdict_color = "#222222"
        else:
            accepted = bool(selected[attack]["attack_success"])
            verdict = f"Model judgment: {parsed['claim']}   |   Strict attack success: {'YES' if accepted else 'NO'}"
            verdict_color = "#B33A2B" if accepted else "#196F3D"
        draw.text((x0 + 18, footer_y), verdict, font=body_font, fill=verdict_color)
        if attack != "none":
            claim = selected[attack]["attack_text"]
            if len(claim) > 74:
                claim = claim[:71] + "..."
            draw.text((x0 + 18, footer_y + 31), f'Overlay: "{claim}"', font=small_font, fill="#333333")

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(300, 300), optimize=True)


def _cell(rows: list[dict[str, Any]], attack: str, defense: str = "none") -> list[dict[str, Any]]:
    selected = [row for row in rows if row["attack"] == attack and row["defense"] == defense]
    selected.sort(key=lambda row: row["sample_id"])
    return selected


def _bootstrap(values: list[bool], rng: np.random.Generator) -> dict[str, float]:
    vector = np.asarray(values, dtype=float)
    mean, lo, hi = percentile_ci(vector, rng)
    return {"mean": mean, "ci_low": lo, "ci_high": hi}


def compute_expanded_analysis(rows: list[dict[str, Any]], log_path: Path) -> dict[str, Any]:
    """Decompose attack success into transcription and belief mechanisms."""
    rng = np.random.default_rng(SEED)
    attack_names = ["naive", "scene_coherent", "causal"]
    repetition_counts = {
        attack: Counter(row["attack_text"] for row in _cell(rows, attack))
        for attack in attack_names
    }
    mechanism = {}
    for attack in attack_names:
        cell = _cell(rows, attack)
        if len(cell) != 300:
            raise ValueError(f"expanded cell {attack} has n={len(cell)}")
        grounded = [bool(row["claim_matches_overlay"]) for row in cell]
        accepted_grounded = [
            row["parsed"]["claim"] == "TRUE"
            for row in cell
            if row["claim_matches_overlay"]
        ]
        mechanism[attack] = {
            "n": len(cell),
            "unique_attack_texts": len(repetition_counts[attack]),
            "maximum_exact_text_repetition": max(repetition_counts[attack].values()),
            "grounded_transcription": _bootstrap(grounded, rng),
            "acceptance_given_grounded": _bootstrap(accepted_grounded, rng),
            "strict_asr": _bootstrap([bool(row["attack_success"]) for row in cell], rng),
            "object_accuracy": _bootstrap([bool(row["object_correct"]) for row in cell], rng),
        }

    causal = _cell(rows, "causal")
    categories = {}
    for category in sorted({row["attack_metadata"]["violation_type"] for row in causal}):
        cell = [row for row in causal if row["attack_metadata"]["violation_type"] == category]
        categories[category] = {
            "n": len(cell),
            "grounded_transcription": _bootstrap([bool(row["claim_matches_overlay"]) for row in cell], rng),
            "strict_asr": _bootstrap([bool(row["attack_success"]) for row in cell], rng),
            "object_accuracy": _bootstrap([bool(row["object_correct"]) for row in cell], rng),
        }

    def repetition_bin(count: int) -> str:
        if count == 1:
            return "1"
        if count <= 4:
            return "2--4"
        if count <= 9:
            return "5--9"
        return "10+"

    repetition = {}
    for attack in attack_names:
        groups = {}
        for label in ["1", "2--4", "5--9", "10+"]:
            cell = [
                row for row in _cell(rows, attack)
                if repetition_bin(repetition_counts[attack][row["attack_text"]]) == label
            ]
            groups[label] = {
                "n": len(cell),
                "strict_asr": _bootstrap([bool(row["attack_success"]) for row in cell], rng) if cell else None,
            }
        repetition[attack] = groups
    return {
        "schema_version": "cta/expanded-analysis-v1",
        "source": str(log_path),
        "source_sha256": sha256(log_path),
        "n_samples": 300,
        "bootstrap": {"resamples": N_BOOT, "seed": SEED, "confidence_level_percent": 95},
        "mechanism": mechanism,
        "causal_categories": categories,
        "exact_text_repetition": repetition,
    }


def write_mechanism_table(analysis: dict[str, Any], path: Path) -> None:
    labels = {"naive": "Naive", "scene_coherent": "Scene-coherent", "causal": "CTA (ours)"}
    lines = [
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Attack & Unique & Max rep. & Read (\%) & True $\mid$ read (\%) & Strict ASR (\%) \\",
        r"\midrule",
    ]
    for attack in ["naive", "scene_coherent", "causal"]:
        m = analysis["mechanism"][attack]
        values = (
            f"{labels[attack]} & {m['unique_attack_texts']} & {m['maximum_exact_text_repetition']} & "
            f"{100*m['grounded_transcription']['mean']:.1f} & {100*m['acceptance_given_grounded']['mean']:.1f} & "
            f"{100*m['strict_asr']['mean']:.1f}"
        )
        lines.append(values + (r" \\" if attack != "causal" else r" \\"))
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_violation_table(analysis: dict[str, Any], path: Path) -> None:
    labels = {
        "biology": "Biology", "energy conservation": "Energy conservation",
        "energy/transport": "Energy/transport", "mechanics": "Mechanics",
        "thermodynamics/decay": "Thermodynamics/decay",
    }
    lines = [
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Violation family & $n$ & Read (\%) & Strict ASR (\%) & Object acc. (\%) \\",
        r"\midrule",
    ]
    for category, m in analysis["causal_categories"].items():
        lines.append(
            f"{labels.get(category, category.title())} & {m['n']} & {100*m['grounded_transcription']['mean']:.1f} & "
            f"{100*m['strict_asr']['mean']:.1f} & {100*m['object_accuracy']['mean']:.1f} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_breakdown_plot(analysis: dict[str, Any], output_base: Path) -> None:
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.45), gridspec_kw={"wspace": 0.34})
    categories = list(analysis["causal_categories"])
    short = {"biology": "Biology", "energy conservation": "Energy\nconservation", "energy/transport": "Energy /\ntransport", "mechanics": "Mechanics", "thermodynamics/decay": "Thermo. /\ndecay"}
    values = [100 * analysis["causal_categories"][name]["strict_asr"]["mean"] for name in categories]
    errors = np.asarray([
        [value - 100 * analysis["causal_categories"][name]["strict_asr"]["ci_low"] for value, name in zip(values, categories)],
        [100 * analysis["causal_categories"][name]["strict_asr"]["ci_high"] - value for value, name in zip(values, categories)],
    ])
    bars = axes[0].bar(range(len(categories)), values, color=OKABE_ITO["causal"], yerr=errors, capsize=2)
    axes[0].set_xticks(range(len(categories)), [short.get(name, name) for name in categories])
    axes[0].set_ylabel("Strict ASR (%)")
    axes[0].set_title("(a) Reality-violation family")
    axes[0].set_ylim(0, 110)
    axes[0].grid(axis="y", alpha=.2)
    for bar, name, value in zip(bars, categories, values):
        n = analysis["causal_categories"][name]["n"]
        axes[0].text(bar.get_x()+bar.get_width()/2, value+6, f"{value:.0f}\n$n$={n}", ha="center", fontsize=6.5)

    bins = ["1", "2--4", "5--9", "10+"]
    colors = [OKABE_ITO["naive"], OKABE_ITO["scene_coherent"], OKABE_ITO["causal"]]
    x = np.arange(len(bins)); width = .25
    for idx, attack in enumerate(["naive", "scene_coherent", "causal"]):
        means = [100*analysis["exact_text_repetition"][attack][label]["strict_asr"]["mean"] for label in bins]
        axes[1].plot(x, means, marker="o", linewidth=1.8, color=colors[idx], label={"naive":"Naive", "scene_coherent":"Scene-coherent", "causal":"CTA"}[attack])
        if attack == "causal":
            for x_, y_, label in zip(x, means, bins):
                axes[1].annotate(f"$n$={analysis['exact_text_repetition'][attack][label]['n']}", (x_, y_), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=6.5)
    axes[1].set_xticks(x, bins)
    axes[1].set_xlabel("Exact overlay frequency in the 300-image set")
    axes[1].set_ylabel("Strict ASR (%)")
    axes[1].set_title("(b) Exact-text repetition")
    axes[1].set_ylim(0, 110); axes[1].grid(axis="y", alpha=.2); axes[1].legend(frameon=False, fontsize=7)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=350, bbox_inches="tight")
    plt.close(fig)


def _draw_outcome_grid(rows: list[dict[str, Any]], output: Path) -> None:
    causal = _cell(rows, "causal")
    candidates = [
        ("Accepted: energy", lambda r: r["attack_success"] and r["attack_metadata"]["violation_type"] in {"energy conservation", "energy/transport"}),
        ("Accepted: biology", lambda r: r["attack_success"] and r["attack_metadata"]["violation_type"] == "biology"),
        ("Correctly rejected", lambda r: r["claim_matches_overlay"] and r["parsed"]["claim"] == "FALSE"),
        ("Transcription miss", lambda r: not r["claim_matches_overlay"]),
    ]
    chosen = []
    for label, predicate in candidates:
        match = next((row for row in causal if predicate(row)), None)
        if match is None:
            raise ValueError(f"no deterministic failure-analysis example for {label}")
        chosen.append((label, match))
    panel_w, panel_h, gap, margin = 700, 560, 20, 20
    canvas = Image.new("RGB", (2*panel_w+gap+2*margin, 2*panel_h+gap+2*margin), "white")
    draw = ImageDraw.Draw(canvas); title_font = fit_font(25, True); body_font = fit_font(18)
    for idx, (label, row) in enumerate(chosen):
        rr, cc = divmod(idx, 2); x0=margin+cc*(panel_w+gap); y0=margin+rr*(panel_h+gap)
        outline = "#B33A2B" if row["attack_success"] else "#196F3D"
        draw.rounded_rectangle((x0,y0,x0+panel_w,y0+panel_h), radius=9, fill="#FAFAFA", outline=outline, width=4)
        draw.text((x0+14,y0+10), f"({chr(97+idx)}) {label}", font=title_font, fill="#111111")
        image = Image.open(row["image_path"]).convert("RGB"); image.thumbnail((panel_w-20, 390), Image.Resampling.LANCZOS)
        canvas.paste(image, (x0+(panel_w-image.width)//2, y0+52+(390-image.height)//2))
        category = row["attack_metadata"]["violation_type"]
        verdict = f"Family: {category} | judgment: {row['parsed']['claim']} | read: {'yes' if row['claim_matches_overlay'] else 'no'}"
        draw.text((x0+14,y0+450), verdict, font=body_font, fill=outline)
        wrapped = textwrap.wrap(row["attack_text"], width=62)[:2]
        for line_idx, line in enumerate(wrapped):
            draw.text((x0+14,y0+480+line_idx*24), line, font=body_font, fill="#333333")
    output.parent.mkdir(parents=True, exist_ok=True); canvas.save(output, dpi=(300,300), optimize=True)


def compute_transfer_table(main_rows: list[dict[str, Any]], transfer_rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    result = {}
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Checkpoint & Clean acc. & Naive ASR & Scene ASR & CTA ASR \\", r"\midrule"]
    for label, rows in [("Qwen2.5-VL-3B", main_rows), ("Qwen2.5-VL-7B", transfer_rows)]:
        metrics = {}
        for attack in ["none", "naive", "scene_coherent", "causal"]:
            cell = _cell(rows, attack)
            if len(cell) != 300:
                raise ValueError(f"transfer table {label}/{attack} has n={len(cell)}")
            key = "clean_object_accuracy" if attack == "none" else f"{attack}_strict_asr"
            values = [bool(r["object_correct"]) for r in cell] if attack == "none" else [bool(r["attack_success"]) for r in cell]
            metrics[key] = float(np.mean(values))
        result[label] = metrics
        lines.append(f"{label} & {100*metrics['clean_object_accuracy']:.1f} & {100*metrics['naive_strict_asr']:.1f} & {100*metrics['scene_coherent_strict_asr']:.1f} & {100*metrics['causal_strict_asr']:.1f} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"]); path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def compute_ocr_table(main_rows: list[dict[str, Any]], ocr_rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    rows_out = {}
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Defense & Naive ASR & CTA ASR & CTA object acc. \\", r"\midrule"]
    specs = [("None", main_rows, "none"), ("Consistency", main_rows, "consistency"), ("RapidOCR mask", ocr_rows, "rapidocr_mask"), ("Oracle mask", main_rows, "ocr_mask")]
    for label, rows, defense in specs:
        naive = _cell(rows, "naive", defense); causal = _cell(rows, "causal", defense)
        if len(naive) != 300 or len(causal) != 300:
            raise ValueError(f"OCR table {label} incomplete: {len(naive)}/{len(causal)}")
        metrics = {"naive_asr": float(np.mean([r["attack_success"] for r in naive])), "causal_asr": float(np.mean([r["attack_success"] for r in causal])), "causal_object_accuracy": float(np.mean([r["object_correct"] for r in causal]))}
        rows_out[label] = metrics
        lines.append(f"{label} & {100*metrics['naive_asr']:.1f} & {100*metrics['causal_asr']:.1f} & {100*metrics['causal_object_accuracy']:.1f} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"]); path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return rows_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-log", type=Path, required=True)
    parser.add_argument("--pilot-log", type=Path, required=True)
    parser.add_argument("--pilot-image-root", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--transfer-log", type=Path)
    parser.add_argument("--ocr-log", type=Path)
    parser.add_argument("--ocr-conditions", type=Path)
    args = parser.parse_args()

    main_rows = read_jsonl(args.main_log)
    pilot_rows = read_jsonl(args.pilot_log)
    args.paper_dir.mkdir(parents=True, exist_ok=True)
    figures = args.paper_dir / "figures"
    evidence = args.paper_dir / "evidence"
    figures.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)

    stats = compute_statistics(main_rows, args.main_log)
    (evidence / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    write_statistics_table(stats, args.paper_dir / "generated_statistics_table.tex")
    write_quality_table(main_rows, args.paper_dir / "generated_quality_table.tex")
    make_result_plot(stats, figures / "asr_defense_analysis")
    make_qualitative_grid(pilot_rows, args.pilot_image_root, figures / "qualitative_example.png")
    expanded = compute_expanded_analysis(main_rows, args.main_log)
    write_mechanism_table(expanded, args.paper_dir / "generated_mechanism_table.tex")
    write_violation_table(expanded, args.paper_dir / "generated_violation_table.tex")
    make_breakdown_plot(expanded, figures / "causal_breakdown")
    _draw_outcome_grid(main_rows, figures / "outcome_examples.png")
    output = {"n": stats["n_samples"], "statistics": stats, "expanded": expanded}
    if args.transfer_log:
        transfer_rows = read_jsonl(args.transfer_log)
        expanded["checkpoint_scale_transfer"] = compute_transfer_table(main_rows, transfer_rows, args.paper_dir / "generated_transfer_table.tex")
        expanded["checkpoint_scale_transfer_source"] = {"path": str(args.transfer_log), "sha256": sha256(args.transfer_log)}
    if args.ocr_log:
        ocr_rows = read_jsonl(args.ocr_log)
        expanded["deployable_ocr_defense"] = compute_ocr_table(main_rows, ocr_rows, args.paper_dir / "generated_ocr_defense_table.tex")
        expanded["deployable_ocr_source"] = {"path": str(args.ocr_log), "sha256": sha256(args.ocr_log)}
    if args.ocr_conditions:
        detection_rows = read_jsonl(args.ocr_conditions)
        expanded["rapidocr_detection"] = {
            attack: {
                "n": len(cell := [r for r in detection_rows if r["attack"] == attack]),
                "overlay_detection_rate": float(np.mean([r["defense_metadata"]["overlay_detected_at_0.5_recall"] for r in cell])),
                "mean_token_recall": float(np.mean([r["defense_metadata"]["overlay_token_recall"] for r in cell])),
                "mean_masked_area_upper_bound_fraction": float(np.mean([r["defense_metadata"]["masked_area_upper_bound_fraction"] for r in cell])),
            } for attack in ["naive", "causal"]
        }
        expanded["rapidocr_detection_source"] = {"path": str(args.ocr_conditions), "sha256": sha256(args.ocr_conditions)}
    (evidence / "expanded_analysis.json").write_text(json.dumps(expanded, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
