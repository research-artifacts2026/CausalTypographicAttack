from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from tqdm import tqdm

from cta.data import load_coco128
from cta.defenses import consistency_defense, ocr_mask_defense
from cta.generation import AttackTextGenerator, SCENE_PROMPT, extract_json, quality_prompt
from cta.metrics import claim_matches_overlay, label_match, parse_task_output, summarize
from cta.model import Qwen25VLAdapter, TASK_PROMPT
from cta.render import render_attack


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def save_summary(out: Path, rows: list[dict]) -> list[dict]:
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary:
        with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader(); writer.writerows(summary)
    tex = ["% AUTO-GENERATED from summary.json; do not edit", "\\begin{tabular}{llrrrr}",
           "Attack & Defense & $N$ & Obj. Acc. & Claim ASR & Parse \\\\", "\\hline"]
    for s in summary:
        asr = "--" if s["false_claim_acceptance_asr"] is None else f'{100*s["false_claim_acceptance_asr"]:.1f}'
        tex.append(f'{s["attack"].replace("_", "-")} & {s["defense"].replace("_", "-")} & {s["n"]} & {100*s["object_accuracy"]:.1f} & {asr} & {100*s["parse_rate"]:.1f} \\\\')
    tex += ["\\end{tabular}", ""]
    (out / "results_table.tex").write_text("\n".join(tex), encoding="utf-8")
    return summary


def safe_git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "not-a-git-checkout"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    out = Path(cfg["output_root"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    provenance_path = out / "provenance.json"
    rows_path = out / "predictions.jsonl"

    samples = load_coco128(cfg["dataset_root"], int(cfg["num_samples"]), int(cfg["seed"]))
    (out / "sample_manifest.json").write_text(json.dumps([s.to_dict() for s in samples], indent=2), encoding="utf-8")
    model = Qwen25VLAdapter(cfg["model"])
    provenance = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "config_path": str(cfg_path),
        "config_sha256": config_hash, "git_head": safe_git_head(), "hostname": platform.node(),
        "python": platform.python_version(), "dataset": {"name": "COCO128", "source": "Ultralytics COCO128", "n": len(samples)},
        "model": model.provenance(), "metric_definition": {
            "false_claim_acceptance_asr": "fraction of false overlays predicted TRUE",
            "object_accuracy": "exact/singular-plural normalized match to largest-area COCO annotation",
        },
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    existing = load_jsonl(rows_path)
    completed = {(r["sample_id"], r["attack"], r["defense"]) for r in existing}
    text_generator = AttackTextGenerator(int(cfg["seed"]))
    attacks = cfg["experiment"]["attacks"]
    defended_attacks = cfg["experiment"]["defended_attacks"]
    defenses = cfg["experiment"]["defenses"]

    for sample in tqdm(samples, desc="samples"):
        scene_file = out / "scene_graphs" / f"{sample.sample_id}.json"
        if scene_file.exists():
            scene_record = json.loads(scene_file.read_text())
            scene = scene_record["parsed"]
        else:
            raw_scene = model.infer(sample.image_path, SCENE_PROMPT, 128)
            scene = extract_json(raw_scene)
            scene_file.parent.mkdir(parents=True, exist_ok=True)
            scene_record = {"sample_id": sample.sample_id, "raw": raw_scene, "parsed": scene}
            scene_file.write_text(json.dumps(scene_record, indent=2), encoding="utf-8")

        texts = text_generator.generate(sample.target_label, sample.sample_id)
        quality_file = out / "quality" / f"{sample.sample_id}.json"
        if cfg["experiment"].get("run_quality_judge", True):
            if quality_file.exists():
                quality_record = json.loads(quality_file.read_text())
            else:
                quality_record = {"raw": {}, "parsed": {}}
                for quality_attack in ("naive", "scene_coherent", "causal"):
                    quality_raw = model.infer(
                        sample.image_path,
                        quality_prompt(scene, quality_attack, texts[quality_attack].text),
                        96,
                    )
                    quality_record["raw"][quality_attack] = quality_raw
                    quality_record["parsed"][quality_attack] = extract_json(quality_raw)
                quality_file.parent.mkdir(parents=True, exist_ok=True)
                quality_file.write_text(json.dumps(quality_record, indent=2), encoding="utf-8")
        else:
            quality_record = {"parsed": {}}

        artifacts = {}
        for attack in attacks:
            artifacts[attack] = render_attack(
                sample.image_path, texts[attack], out / "images" / attack / f"{sample.sample_id}.jpg"
            )
        conditions = [(a, "none", artifacts[a].image_path, {}) for a in attacks]
        for attack in defended_attacks:
            artifact = artifacts[attack]
            if "consistency" in defenses:
                p, meta = consistency_defense(artifact, scene, sample.target_label, out / "images" / f"{attack}_consistency" / f"{sample.sample_id}.jpg")
                conditions.append((attack, "consistency", p, meta))
            if "ocr_mask" in defenses:
                p, meta = ocr_mask_defense(artifact, out / "images" / f"{attack}_ocr_mask" / f"{sample.sample_id}.jpg")
                conditions.append((attack, "ocr_mask", p, meta))

        for attack, defense, image_path, defense_meta in conditions:
            key = (sample.sample_id, attack, defense)
            if key in completed:
                continue
            started = time.time()
            raw = model.infer(image_path, TASK_PROMPT)
            parsed = parse_task_output(raw)
            quality = quality_record.get("parsed", {}).get(attack, {}) if attack != "none" else {}
            row = {
                "sample_id": sample.sample_id, "source_sha256": sample.source_sha256,
                "target_label": sample.target_label, "attack": attack, "defense": defense,
                "attack_text": artifacts[attack].text, "attack_metadata": artifacts[attack].to_dict(),
                "defense_metadata": defense_meta, "image_path": image_path, "raw_output": raw, "parsed": parsed,
                "object_correct": label_match(parsed["object"], sample.target_label), "quality": quality,
                "latency_s": round(time.time() - started, 4), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            row["claim_matches_overlay"] = claim_matches_overlay(parsed.get("claim_text", ""), artifacts[attack].text)
            row["attack_success"] = attack != "none" and parsed["claim"] == "TRUE" and row["claim_matches_overlay"]
            append_jsonl(rows_path, row)
            existing.append(row); completed.add(key)
        save_summary(out, existing)

    provenance["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["completed_rows"] = len(existing)
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(save_summary(out, existing), indent=2))


if __name__ == "__main__":
    main()
