#!/usr/bin/env python3
"""Create deterministic qualitative examples from complete budgeted-attack logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.reality_violation import exact_transcription_match


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be MODEL=RUN_DIR")
    model, path = value.split("=", 1)
    return model, Path(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def status(row: dict) -> str:
    parsed = row.get("parsed", {})
    exact = bool(
        parsed.get("parse_ok")
        and exact_transcription_match(parsed.get("claim_text", ""), row.get("attack_text", ""))
    )
    if exact and str(parsed.get("claim", "")).upper() == "TRUE":
        return "T"
    if exact:
        return "R"
    return "M"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--policy-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) != 4:
        raise ValueError("qualitative figure requires exactly four model runs")
    selection = json.loads(args.policy_file.resolve().read_text(encoding="utf-8"))
    policies = selection["selected_policy_ids"]
    rank = {policy: index for index, policy in enumerate(policies)}
    model_order = [model for model, _ in args.run]
    rows_by_model = {}
    evidence = []
    for model, run_dir in args.run:
        path = run_dir.resolve() / "predictions.jsonl"
        rows = [row for row in read_jsonl(path) if row["attack"] in rank]
        if len(rows) != 80 * len(policies):
            raise ValueError(f"{model} run is incomplete")
        rows_by_model[model] = {(row["sample_id"], row["attack"]): row for row in rows}
        evidence.append({"model": model, "path": str(path), "sha256": sha256(path)})
    keys = sorted(rows_by_model[model_order[0]], key=lambda key: (key[0], rank[key[1]]))
    if any(set(rows_by_model[model]) != set(keys) for model in model_order):
        raise ValueError("model logs do not share the same sample-policy universe")

    def statuses(key: tuple[str, str]) -> dict[str, str]:
        return {model: status(rows_by_model[model][key]) for model in model_order}

    broad = sorted(keys, key=lambda key: (-sum(value == "T" for value in statuses(key).values()), key[0], rank[key[1]]))[0]
    qwen_only_candidates = [
        key for key in keys
        if statuses(key)[model_order[0]] == "T"
        and statuses(key)[model_order[1]] == "T"
        and statuses(key)[model_order[2]] != "T"
        and statuses(key)[model_order[3]] != "T"
        and key != broad
    ]
    qwen_only = qwen_only_candidates[0] if qwen_only_candidates else next(key for key in keys if key != broad)
    cross_candidates = [
        key for key in keys
        if (statuses(key)[model_order[2]] == "T" or statuses(key)[model_order[3]] == "T")
        and key[0] not in {broad[0], qwen_only[0]}
    ]
    if not cross_candidates:
        cross_candidates = [
            key for key in keys
            if (statuses(key)[model_order[2]] == "T" or statuses(key)[model_order[3]] == "T")
            and key not in {broad, qwen_only}
        ]
    cross = sorted(
        cross_candidates,
        key=lambda key: (-sum(value == "T" for value in statuses(key).values()), key[0], rank[key[1]]),
    )[0]
    chosen = [("maximum shared success", broad), ("Qwen-only transfer", qwen_only), ("cross-family success", cross)]

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.65))
    records = []
    for axis, (role, key) in zip(axes, chosen):
        reference = rows_by_model[model_order[0]][key]
        image_path = Path(reference["image_path"])
        axis.imshow(Image.open(image_path).convert("RGB"))
        axis.axis("off")
        outcome = statuses(key)
        axis.set_title(
            role + "\n" + key[0] + "\n" + "  ".join(f"{model}: {outcome[model]}" for model in model_order),
            fontsize=7.5,
        )
        records.append({
            "role": role,
            "sample_id": key[0],
            "policy_id": key[1],
            "attack_text": reference["attack_text"],
            "image_path": str(image_path),
            "image_sha256": sha256(image_path),
            "status_by_model": outcome,
        })
    figure.text(0.5, 0.015, "T = exact transcription + TRUE; R = exact transcription + rejection; M = transcription miss", ha="center")
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    provenance = {
        "schema_version": "cta/budgeted-qualitative-figure-v1",
        "selection_rule": "first deterministic key satisfying each logged outcome role; no aesthetic selection",
        "policy_file": str(args.policy_file.resolve()),
        "policy_file_sha256": sha256(args.policy_file.resolve()),
        "evidence_files": evidence,
        "examples": records,
    }
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "examples": records}, indent=2))


if __name__ == "__main__":
    main()
