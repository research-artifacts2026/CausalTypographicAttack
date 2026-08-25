#!/usr/bin/env python3
"""Generate model-rerating and cross-OCR paper assets from immutable summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


METHOD_LABELS = {
    "naive": "Naive",
    "scene_coherent": "Scene-coherent",
    "causal_compact_pil": "Causal (PIL)",
    "causal_compact_textdiffuser": "Causal (TextDiffuser)",
}

METRICS = (
    ("legibility_1to5", "Legibility"),
    ("visual_integration_1to5", "Integration"),
    ("scene_fit_1to5", "Scene fit"),
    ("impossibility_1to5", "Impossible"),
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("_", r"\_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind-results", type=Path, required=True)
    parser.add_argument("--blind-provenance", type=Path, required=True)
    parser.add_argument("--rapid-qwen3", type=Path, required=True)
    parser.add_argument("--rapid-qwen7", type=Path, required=True)
    parser.add_argument("--easy-qwen3", type=Path, required=True)
    parser.add_argument("--easy-qwen7", type=Path, required=True)
    parser.add_argument("--easy-mask-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        key: value.resolve()
        for key, value in vars(args).items()
        if key != "output_dir"
    }
    blind = load(paths["blind_results"])
    blind_provenance = load(paths["blind_provenance"])
    if blind.get("evaluator_kind") != "model" or blind.get("evaluator_model") != "gpt-5.6-sol":
        raise ValueError("blind result is not labeled as GPT-5.6-sol model evaluation")
    if blind_provenance.get("prohibited_label") != "three independent human annotators":
        raise ValueError("blind provenance does not preserve the human/model boundary")
    expected_result_hash = blind_provenance.get("result_sha256", "").lower()
    if expected_result_hash != sha256(paths["blind_results"]):
        raise ValueError("blind result hash does not match provenance")

    ocr_sources = (
        ("RapidOCR", "Qwen2.5-VL-3B", paths["rapid_qwen3"]),
        ("RapidOCR", "Qwen2.5-VL-7B", paths["rapid_qwen7"]),
        ("EasyOCR", "Qwen2.5-VL-3B", paths["easy_qwen3"]),
        ("EasyOCR", "Qwen2.5-VL-7B", paths["easy_qwen7"]),
    )
    ocr_rows = []
    for engine, model, path in ocr_sources:
        summary = load(path)
        style = summary["styles"][0]
        if summary.get("parse_failure_successes") != 0:
            raise ValueError(f"{path}: parse failure counted as success")
        ocr_rows.append({
            "engine": engine,
            "model": model,
            "samples": summary["samples"],
            "clean_eligible_n": style["clean_eligible_n"],
            "strict_successes": style["strict_successes"],
            "strict_asr": style["strict_asr"],
            "complete_transcription_rate": style["complete_transcription_rate"],
            "mean_detector_token_recall": style["mean_detector_token_recall"],
            "mean_carrier_survival_fraction": style["mean_carrier_survival_fraction"],
            "summary_path": str(path),
            "summary_sha256": sha256(path),
            "prediction_log_sha256": summary["prediction_log_sha256"],
            "clean_log_sha256": summary["clean_log_sha256"],
        })
    easy_mask = load(paths["easy_mask_provenance"])
    if easy_mask.get("status") != "complete" or easy_mask.get("rows") != 20:
        raise ValueError("EasyOCR mask provenance is incomplete")

    evidence = {
        "schema_version": "cta/gap-closing-paper-evidence-v1",
        "blind_model_evaluation": blind,
        "blind_model_provenance": blind_provenance,
        "blind_result_sha256": sha256(paths["blind_results"]),
        "blind_provenance_sha256": sha256(paths["blind_provenance"]),
        "secondary_ocr": {
            "rows": ocr_rows,
            "easy_mask_provenance": easy_mask,
            "easy_mask_provenance_sha256": sha256(paths["easy_mask_provenance"]),
            "claim_boundary": "Twenty-image detector-transfer diagnostic with small clean-eligible denominators; not a broad cross-OCR robustness claim.",
        },
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "gap_closing_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    blind_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        "Method & " + " & ".join(label for _, label in METRICS) + r" \\",
        r"\midrule",
    ]
    for method in ("naive", "scene_coherent", "causal_compact_pil", "causal_compact_textdiffuser"):
        values = blind["methods"][method]
        blind_lines.append(
            tex_escape(METHOD_LABELS[method]) + " & "
            + " & ".join(f"{values[key]['mean']:.2f}" for key, _ in METRICS)
            + r" \\"
        )
    blind_lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    (output / "generated_gpt56sol_blind_table.tex").write_text("\n".join(blind_lines), encoding="utf-8")

    ocr_lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Mask engine & Victim & $n_c$ & OCR recall & Carrier left & Read & ASR \\",
        r"\midrule",
    ]
    for row in ocr_rows:
        ocr_lines.append(
            "{} & {} & {} & {:.1f} & {:.1f} & {:.1f} & {:.1f} \\\\".format(
                row["engine"], row["model"], row["clean_eligible_n"],
                100 * row["mean_detector_token_recall"],
                100 * row["mean_carrier_survival_fraction"],
                100 * row["complete_transcription_rate"],
                100 * row["strict_asr"],
            )
        )
    ocr_lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    (output / "generated_secondary_ocr_table.tex").write_text("\n".join(ocr_lines), encoding="utf-8")
    print(json.dumps({"blind_methods": len(blind["methods"]), "ocr_rows": len(ocr_rows)}, indent=2))


if __name__ == "__main__":
    main()
