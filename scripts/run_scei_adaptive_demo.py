#!/usr/bin/env python3
"""Run one bounded black-box SCEI session without the Gradio front end."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.model import build_model_adapter
from cta.scei_adaptive import adaptive_scei_events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--target-label", default="")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--renderer", choices=("scene", "flat"), default="scene")
    parser.add_argument("--no-strict-read-gate", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    planner_cfg = config["planner_model"]
    victim_cfg = config["victim_model"]
    planner = build_model_adapter(planner_cfg)
    victim = planner if planner_cfg == victim_cfg else build_model_adapter(victim_cfg)
    for event in adaptive_scei_events(
        args.image.resolve(),
        args.target_label,
        planner,
        victim,
        args.output_root.resolve(),
        max_rounds=args.max_rounds,
        renderer_mode=args.renderer,
        strict_read_gate=not args.no_strict_read_gate,
        max_planner_attempts=int(config.get("max_planner_attempts", 3)),
    ):
        print(json.dumps({
            "stage": event["stage"],
            "round": event["round"],
            "image_path": event["image_path"],
            "target_label": event["target_label"],
            "answer_raw": event["answer_raw"],
            "parsed_semantic": event["parsed_semantic"],
            "exact_read_match": event.get("exact_read_match"),
            "success": event["success"],
            "design": event.get("design"),
        }, ensure_ascii=False), flush=True)
    summary_path = args.output_root.resolve() / "summary.json"
    if summary_path.is_file():
        print(summary_path.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
