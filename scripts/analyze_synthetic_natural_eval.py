#!/usr/bin/env python3
"""Validate and aggregate the four-model synthetic-natural n=3 pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import file_sha256


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--model-log", action="append", required=True, help="MODEL=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    registry_path = args.registry.resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    items = {item["item_id"]: item for item in registry["items"]}
    expected = {(item_id, condition) for item_id in items for condition in ("no_attack", "synthetic_natural")}
    evidence = {
        "schema_version": "cta/synthetic-natural-crossmodel-analysis-v1",
        "status": "complete",
        "evidence_label": "synthetic natural-render; not physical capture",
        "registry": str(registry_path),
        "registry_sha256": file_sha256(registry_path),
        "items": len(items),
        "warning": "n=3 qualitative feasibility pilot; never report as a headline, physical result, or SOTA",
        "models": {},
    }
    table_rows = []
    for assignment in args.model_log:
        if "=" not in assignment:
            raise ValueError("--model-log must use MODEL=PATH")
        model_name, value = assignment.split("=", 1)
        log_path = Path(value).resolve()
        rows = read_jsonl(log_path)
        keys = {(row["item_id"], row["condition"]) for row in rows}
        if keys != expected or len(rows) != len(expected):
            raise ValueError(f"{model_name}: incomplete or duplicate log")
        for row in rows:
            item = items[row["item_id"]]
            expected_hash = item["source_sha256"] if row["condition"] == "no_attack" else item["image_sha256"]
            if row["image_sha256"] != expected_hash or file_sha256(row["image_path"]) != expected_hash:
                raise ValueError(f"{model_name}: image hash mismatch for {(row['item_id'], row['condition'])}")
        summary_path = log_path.parent / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete" or summary.get("items") != len(items):
            raise ValueError(f"{model_name}: incomplete summary")
        clean = {row["item_id"]: row for row in rows if row["condition"] == "no_attack"}
        attacked = {row["item_id"]: row for row in rows if row["condition"] == "synthetic_natural"}
        eligible = [item_id for item_id, row in clean.items() if row["answer_correct"]]
        successes = [item_id for item_id in eligible if attacked[item_id]["target_match"]]
        grounded = [item_id for item_id in successes if attacked[item_id]["read_match"]]
        read_count = sum(attacked[item_id]["read_match"] for item_id in items)
        model_result = {
            "log": str(log_path),
            "log_sha256": file_sha256(log_path),
            "n_clean_correct": len(eligible),
            "target_successes": len(successes),
            "grounded_successes": len(grounded),
            "attacked_read_successes": read_count,
            "successful_item_ids": successes,
            "grounded_successful_item_ids": grounded,
        }
        evidence["models"][model_name] = model_result
        table_rows.append((model_name, model_result))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "synthetic_natural_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "\\begin{tabular}{lrrrr}", "\\toprule",
        "Model & Clean & Read & Target & Grounded \\\\", "\\midrule",
    ]
    for model_name, row in table_rows:
        lines.append(
            f"{model_name} & {row['n_clean_correct']}/3 & {row['attacked_read_successes']}/3 & "
            f"{row['target_successes']}/{row['n_clean_correct']} & "
            f"{row['grounded_successes']}/{row['n_clean_correct']} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    (output / "generated_synthetic_natural_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"models": list(evidence["models"]), "items": len(items)}, indent=2))


if __name__ == "__main__":
    main()

