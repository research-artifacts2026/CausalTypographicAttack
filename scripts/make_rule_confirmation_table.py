#!/usr/bin/env python3
"""Recompute the prospective rule-explicit comparison from audited pair vectors."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def generate(evidence: Path, output: Path):
    data = json.loads(evidence.read_text(encoding="utf-8"))
    if data["status"] != "complete" or data["actual_victim_calls"] != 1280:
        raise ValueError("requires the complete registered 1280-call study")
    if len(data["results"]) != 2 or {r["model"] for r in data["results"]} != {"qwen7", "qwen3vl8"}:
        raise ValueError("both registered models must be retained")
    tests = {t["model"]: t for t in data["primary_tests"]}
    recomputed = []
    for row in data["results"]:
        if row["items"] != 128 or row["runtime_errors"]:
            raise ValueError("incomplete/error-containing study")
        subset = row["clean_correct_subset"]
        a, b = [subset[c]["target_vector"] for c in ("simple_false", "rule_false")]
        n = row["clean_correct"]
        if len(a) != n or len(b) != n or any(v not in (0, 1) for v in a + b):
            raise ValueError("invalid paired target vectors")
        plus = sum(x == 0 and y == 1 for x, y in zip(a, b))
        minus = sum(x == 1 and y == 0 for x, y in zip(a, b))
        discordant = plus + minus
        p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(plus, minus) + 1)) / 2**discordant) if discordant else 1.
        t = tests[row["model"]]
        if not math.isclose(p, t["p"]) or plus != t["plus"] or minus != t["minus"]:
            raise ValueError("paired test disagrees with vectors")
        for values, condition in ((a, "simple_false"), (b, "rule_false")):
            if sum(values) != subset[condition]["target_count"]:
                raise ValueError("target count mismatch")
        recomputed.append({"model": row["model"], "n": n, "simple": sum(a), "rule": sum(b),
            "delta": (sum(b) - sum(a)) / n, "p": p, "plus": plus, "minus": minus,
            "simple_valid": row["all_scene"]["simple_true"]["correct_count"],
            "rule_valid": row["all_scene"]["rule_true"]["correct_count"]})
    maximum = 0.
    for index, r in enumerate(sorted(recomputed, key=lambda x: x["p"])):
        maximum = max(maximum, min(1., (2 - index) * r["p"]))
        r["holm_p"] = maximum
        if not math.isclose(maximum, tests[r["model"]]["holm_p"]):
            raise ValueError("Holm result mismatch")
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Model & Clean & Simple false & Rule false & $\Delta$ (pp) & $p_H$ & Simple valid & Rule valid \\", r"\midrule"]
    names = {"qwen7": "Qwen2.5-VL-7B", "qwen3vl8": "Qwen3-VL-8B"}
    for r in recomputed:
        ptext = f"{r['holm_p']:.6f}" if r["holm_p"] < .01 else f"{r['holm_p']:.3f}"
        lines.append(f"{names[r['model']]} & {r['n']}/128 & {r['simple']}/{r['n']} & {r['rule']}/{r['n']} & {r['delta']*100:+.2f} & {ptext} & {r['simple_valid']}/128 & {r['rule_valid']}/128 " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "generated_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    record = {"input_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
              "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "checks": "paired vectors, exact McNemar, Holm and all registered models retained",
              "results": recomputed, "claim_boundary": data["limits"]}
    (output / "table_provenance.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.evidence, args.output)
