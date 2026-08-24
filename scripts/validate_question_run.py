#!/usr/bin/env python3
"""Refuse table generation unless a paired question run is complete."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.run_validation import validate_question_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"]).resolve()
    audit = validate_question_run(
        Path(config["source_manifest"]).resolve(),
        output_root / "predictions.jsonl",
        output_root / "provenance.json",
        expected_questions=int(config.get("expected_questions", 0)),
        config_path=config_path,
    )
    text = json.dumps(audit, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
