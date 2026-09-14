#!/usr/bin/env python3
"""Freeze, verify, or evaluate an immutable three-state packet."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cta.verification_workbench import (STRATEGIES, SUPPORTED_STRATEGIES, evaluate_packet, freeze_packet,
    load_packet, read_jsonl, select_items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--manifest", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--items", type=int, required=True)
    freeze.add_argument("--strategies", nargs="+", choices=SUPPORTED_STRATEGIES, default=list(STRATEGIES))
    verify = sub.add_parser("verify")
    verify.add_argument("--packet", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--packet", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        chosen = select_items(read_jsonl(args.manifest), args.items)
        print(freeze_packet(chosen, args.output, origin="frozen-manifest diagnostic subset", strategies=args.strategies))
    elif args.action == "verify":
        packet, rows = load_packet(args.packet)
        print(json.dumps({"status": "passed", "items": packet["items"], "rows": len(rows)}))
    else:
        from cta.model import build_model_adapter
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))["model"]
        result = evaluate_packet(args.packet, args.output, cfg, build_model_adapter,
            lambda n, key: print(f"{n} calls recorded: {key}", flush=True))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
