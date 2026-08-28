#!/usr/bin/env python3
"""Generate paper tables only from complete, provenance-linked extension logs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pct(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.1f}"


def condition(summary: dict, name: str) -> dict:
    return next(row for row in summary["conditions"] if row["condition"] == name)


def macro(summary: dict, name: str) -> float:
    values = [
        next(row for row in stratum["conditions"] if row["condition"] == name)["grounded_clean_conditioned_asr"]
        for stratum in summary["strata"]
    ]
    if len(values) != 6 or any(value is None for value in values):
        raise ValueError(f"balanced summary lacks six valid cells for {name}")
    return sum(values) / len(values)


def assert_complete(provenance: dict, expected_rows: int) -> None:
    if provenance.get("status") != "complete" or int(provenance.get("completed_rows", -1)) != expected_rows:
        raise ValueError(f"incomplete run provenance: expected {expected_rows} rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)

    core_paths = {
        "sem_qwen_summary": root / "semantic_defense_qwen7_coco_n300" / "summary.json",
        "sem_qwen_prov": root / "semantic_defense_qwen7_coco_n300" / "provenance.json",
        "sem_llava_summary": root / "semantic_defense_llava_coco_n300" / "summary.json",
        "sem_llava_prov": root / "semantic_defense_llava_coco_n300" / "provenance.json",
        "render_qwen_summary": root / "matched_bridge_renderer_qwen7_coco_n300" / "summary.json",
        "render_qwen_prov": root / "matched_bridge_renderer_qwen7_coco_n300" / "provenance.json",
        "render_llava_summary": root / "matched_bridge_renderer_llava_coco_n300" / "summary.json",
        "render_llava_prov": root / "matched_bridge_renderer_llava_coco_n300" / "provenance.json",
    }
    optional_paths = {
        "qwen3_summary": root / "rvtaqa_balanced_coco_qwen3vl8_n300" / "summary.json",
        "qwen3_prov": root / "rvtaqa_balanced_coco_qwen3vl8_n300" / "provenance.json",
        "internvl3_summary": root / "rvtaqa_balanced_coco_internvl3_8b_n300" / "summary.json",
        "internvl3_prov": root / "rvtaqa_balanced_coco_internvl3_8b_n300" / "provenance.json",
    }
    missing = [str(path) for path in core_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing extension evidence: " + ", ".join(missing))
    optional_exists = all(path.is_file() for path in optional_paths.values())
    paths = {**core_paths, **(optional_paths if optional_exists else {})}
    values = {name: load(path) for name, path in paths.items()}
    for prefix in ("sem_qwen", "sem_llava", "render_qwen", "render_llava"):
        assert_complete(values[f"{prefix}_prov"], 900)
    if optional_exists:
        assert_complete(values["qwen3_prov"], 1800)
        assert_complete(values["internvl3_prov"], 1800)

    defense_rows = []
    for model, prefix in (("Qwen2.5-VL-7B", "sem_qwen"), ("LLaVA-OV-8B", "sem_llava")):
        summary = values[f"{prefix}_summary"]
        paired = summary["paired_bridge"]
        defense_rows.append(
            f"{model} & {summary['base_clean_correct']}/300 & {summary['defense_clean_correct']}/300 & "
            f"{pct(paired['base_grounded_asr'])} & {pct(paired['defense_grounded_asr'])} & "
            f"{100 * (paired['defense_grounded_asr'] - paired['base_grounded_asr']):+.1f} & "
            f"{paired['exact_mcnemar_p']:.2g} \\\\"
        )
    defense_tex = "\n".join([
        "\\begin{tabular}{lrrrrrr}", "\\toprule",
        "Model & Base clean & Def. clean & Bridge & Defended & $\\Delta$ASR & $p$ \\\\",
        "\\midrule", *defense_rows, "\\bottomrule", "\\end{tabular}", "",
    ])
    (output / "generated_semantic_world_defense_table.tex").write_text(defense_tex, encoding="utf-8")

    renderer_rows = []
    for model, prefix in (("Qwen2.5-VL-7B", "render_qwen"), ("LLaVA-OV-8B", "render_llava")):
        summary = values[f"{prefix}_summary"]
        paired = summary["paired_flat_vs_scene"]
        renderer_rows.append(
            f"{model} & {summary['clean_correct']} & {pct(paired['flat_grounded_asr'])} & "
            f"{pct(paired['scene_grounded_asr'])} & "
            f"{100 * (paired['scene_grounded_asr'] - paired['flat_grounded_asr']):+.1f} & "
            f"{paired['exact_mcnemar_p']:.2g} \\\\"
        )
    renderer_tex = "\n".join([
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        "Model & $n_c$ & Flat & Scene-integrated & $\\Delta$ & $p$ \\\\",
        "\\midrule", *renderer_rows, "\\bottomrule", "\\end{tabular}", "",
    ])
    (output / "generated_matched_bridge_renderer_table.tex").write_text(renderer_tex, encoding="utf-8")

    generated_outputs = [
        "generated_semantic_world_defense_table.tex",
        "generated_matched_bridge_renderer_table.tex",
    ]
    if optional_exists:
        model_rows = []
        for model, prefix in (("Qwen3-VL-8B", "qwen3"), ("InternVL3-8B", "internvl3")):
            summary = values[f"{prefix}_summary"]
            clean = next(row for row in summary["pooled"] if row["condition"] == "no_attack")["answer_accuracy"]
            model_rows.append(
                f"{model} & 300 & {pct(clean)} & {pct(macro(summary, 'plain_claim'))} & "
                f"{pct(macro(summary, 'evidence_cta'))} & {pct(macro(summary, 'causal_bridge'))} \\\\"
            )
        model_tex = "\n".join([
            "\\begin{tabular}{lrrrrr}", "\\toprule",
            "Model & $n$ & Clean & Plain & Evidence & Bridge-M \\\\",
            "\\midrule", *model_rows, "\\bottomrule", "\\end{tabular}", "",
        ])
        (output / "generated_new_model_extension_table.tex").write_text(model_tex, encoding="utf-8")
        generated_outputs.append("generated_new_model_extension_table.tex")

    ledger = {
        "schema_version": "cta/paper-extension-evidence-v1",
        "inputs": {
            name: {"path": str(path.relative_to(root)), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "outputs": {
            name: {"path": name, "sha256": sha256(output / name)}
            for name in generated_outputs
        },
        "claim_boundary": {
            "renderer": "same registered text and geometry; deterministic synthetic scene integration, not camera capture",
            "defense": "attack-oblivious prompt wrapper; reports clean utility cost and paired ASR",
            "models": (
                "Qwen3-VL and InternVL3 are both n=300"
                if optional_exists else "new-model tables withheld until both n=300 runs complete"
            ),
        },
    }
    (output / "extension_evidence_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
