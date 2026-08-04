import argparse
import json
from pathlib import Path

from cta.metrics import summarize
from run_experiment import load_jsonl, save_summary

parser = argparse.ArgumentParser()
parser.add_argument("run_dir")
parser.add_argument("--copy-to", default=None)
args = parser.parse_args()
run_dir = Path(args.run_dir)
summary = save_summary(run_dir, load_jsonl(run_dir / "predictions.jsonl"))
if args.copy_to:
    Path(args.copy_to).write_text((run_dir / "results_table.tex").read_text())
print(json.dumps(summary, indent=2))

