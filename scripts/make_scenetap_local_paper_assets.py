#!/usr/bin/env python3
"""Generate the local-planner SceneTAP diagnostic table from official scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_evidence(
    plan_path: Path, render_path: Path, eval_path: Path, scores: list[tuple[str, Path]],
) -> dict:
    plan = load(plan_path)
    render = load(render_path)
    evaluation = load(eval_path)
    for name, value in (("plan", plan), ("render", render), ("evaluation", evaluation)):
        if value.get("status") != "complete":
            raise ValueError(f"{name} provenance is not complete")
    plans_jsonl = Path(plan["plans_path"])
    if not plans_jsonl.is_file():
        raise FileNotFoundError(f"plan provenance points to missing JSONL: {plans_jsonl}")
    plan_rows = read_jsonl(plans_jsonl)
    if len(plan_rows) != int(plan["questions"]):
        raise ValueError("plan provenance question count differs from plans JSONL")
    planning_audit = {
        "rows": len(plan_rows),
        "plans_jsonl": str(plans_jsonl),
        "plans_jsonl_sha256": sha256(plans_jsonl),
        "region_fallbacks": sum(
            bool(row.get("region_resolution", {}).get("used_fallback")) for row in plan_rows
        ),
        "caption_fallbacks": sum(
            bool(row.get("caption_resolution", {}).get("used_fallback")) for row in plan_rows
        ),
        "resolved_captions_missing_attack_text": sum(
            row["adversarial_text"].casefold() not in row["plan"]["short_caption"].casefold()
            for row in plan_rows
        ),
    }
    models = {}
    for model, path in scores:
        score = load(path)
        conditions = score.get("conditions", {})
        required = {"no_attack", "scenetap_full_local_qwen_planner"}
        if not required <= set(conditions):
            raise ValueError(f"{model}: official score is missing {sorted(required - set(conditions))}")
        clean = conditions["no_attack"]
        attack = conditions["scenetap_full_local_qwen_planner"]
        if clean["n"] != attack["n"] or clean["n_clean_correct"] != attack["n_clean_correct"]:
            raise ValueError(f"{model}: clean/attack denominator mismatch")
        models[model] = {
            "official_score_path": str(path),
            "official_score_sha256": sha256(path),
            "official_code_commit": score.get("official_code_commit"),
            "questions": attack["n"],
            "n_clean_correct": attack["n_clean_correct"],
            "clean_accuracy": clean["accuracy"],
            "clean_conditioned_asr": attack["clean_conditioned_asr"],
            "attacked_accuracy": attack["accuracy"],
        }
    return {
        "schema_version": "cta/scenetap-local-qwen-paper-evidence-v1",
        "method_label": "SceneTAP full chain (local Qwen planner)",
        "official_equivalence": False,
        "boundary": "Official SoM and TextDiffuser components with a local Qwen2.5-VL-7B planner; not the official GPT-4o planner service.",
        "planning": {
            "path": str(plan_path), "sha256": sha256(plan_path),
            "audit": planning_audit, **plan,
        },
        "rendering": {"path": str(render_path), "sha256": sha256(render_path), **render},
        "evaluation_manifest": {"path": str(eval_path), "sha256": sha256(eval_path), **evaluation},
        "models": models,
    }


def make_table(evidence: dict) -> str:
    rows = []
    for model, row in evidence["models"].items():
        rows.append(
            f"{model} & {row['n_clean_correct']} & {100 * row['clean_accuracy']:.1f} & "
            f"{100 * row['attacked_accuracy']:.1f} & {100 * row['clean_conditioned_asr']:.1f} \\\\"
        )
    return "\n".join([
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Victim & $n_c$ & Clean acc. & Attack acc. & ASR \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-provenance", type=Path, required=True)
    parser.add_argument("--render-provenance", type=Path, required=True)
    parser.add_argument("--eval-provenance", type=Path, required=True)
    parser.add_argument("--score", action="append", required=True, help="MODEL=official_rio_score.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    scores = []
    for assignment in args.score:
        if "=" not in assignment:
            raise ValueError("--score must use MODEL=PATH")
        model, path = assignment.split("=", 1)
        scores.append((model, Path(path).resolve()))
    evidence = build_evidence(
        args.plan_provenance.resolve(), args.render_provenance.resolve(),
        args.eval_provenance.resolve(), scores,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scenetap_local_qwen_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "generated_scenetap_local_qwen_table.tex").write_text(
        make_table(evidence), encoding="utf-8"
    )
    print(json.dumps({"models": list(evidence["models"]), "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
