#!/usr/bin/env python3
"""Pair a locally planned SceneTAP render with its frozen clean RIO rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_rows(base_rows: list[dict], rendered_rows: list[dict]) -> list[dict]:
    clean = {
        str(row["question_id"]): row for row in base_rows
        if row.get("condition") == "no_attack"
    }
    if len(clean) != len({str(row["question_id"]) for row in clean.values()}):
        raise ValueError("duplicate clean question ids")
    result: list[dict] = []
    seen: set[str] = set()
    for rendered in rendered_rows:
        question_id = str(rendered["question_id"])
        if question_id in seen:
            raise ValueError(f"duplicate rendered question id: {question_id}")
        if question_id not in clean:
            raise KeyError(f"no frozen clean row for question id {question_id}")
        seen.add(question_id)
        clean_row = dict(clean[question_id])
        attacked = dict(clean_row)
        attacked.update({
            "condition": "scenetap_full_local_qwen_planner",
            "image_path": rendered["image_path"],
            "image_sha256": rendered["image_sha256"],
            "overlay_text": rendered["adversarial_text"],
            "attack_word": rendered["adversarial_text"],
            "base_image_path": clean_row["image_path"],
            "base_image_sha256": clean_row["image_sha256"],
            "official_attack_metadata": {
                "pipeline": "official SoM + local Qwen2.5-VL-7B planner + official TextDiffuser",
                "official_equivalence": False,
                "selected_candidate_index": rendered["selected_candidate_index"],
                "candidate_count": rendered["candidate_count"],
                "bbox": rendered["bbox"],
                "region_resolution": rendered.get("region_resolution"),
                "render_manifest_schema": rendered.get("schema_version"),
            },
        })
        result.extend((clean_row, attacked))
    expected = 2 * len(rendered_rows)
    if len(result) != expected:
        raise AssertionError(f"expected {expected} paired rows, got {len(result)}")
    return sorted(result, key=lambda row: (str(row["question_id"]), row["condition"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    manifest = output_root / "render_manifest.jsonl"
    if manifest.exists():
        raise FileExistsError(f"refusing to overwrite {manifest}")
    base_manifest = args.base_manifest.resolve()
    render_manifest = args.render_manifest.resolve()
    rows = build_rows(read_jsonl(base_manifest), read_jsonl(render_manifest))
    output_root.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    provenance = {
        "schema_version": "cta/scenetap-local-qwen-eval-manifest-v1",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_manifest": str(base_manifest),
        "base_manifest_sha256": sha256(base_manifest),
        "render_manifest": str(render_manifest),
        "render_manifest_sha256": sha256(render_manifest),
        "questions": len(rows) // 2,
        "conditions": ["no_attack", "scenetap_full_local_qwen_planner"],
        "rows": len(rows),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "official_equivalence": False,
    }
    (output_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
