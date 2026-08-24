#!/usr/bin/env python3
"""Materialize a paired RIO Obj-MC pilot plus question-conditioned CTA images."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import CONDITIONS, build_spec, file_sha256, render_condition
from cta.rio_bench import (
    RIO_CONFIG_GROUPS, RIO_CONDITION_BY_CONFIG, stable_reservoir,
    target_letter_from_attack_word,
)


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def load_stream(repo_id: str, config: str, split: str, revision: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install `datasets` before materializing RIO-Bench") from exc
    try:
        return load_dataset(
            repo_id, config, split=split, revision=revision, streaming=True,
        )
    except ValueError as exc:
        # The Hub stores each advertised config in a top-level data directory,
        # while some `datasets` releases expose only a synthetic `default`
        # BuilderConfig. Loading the same pinned files by data_dir preserves
        # config and revision semantics without mixing repository subsets.
        if "BuilderConfig" not in str(exc) or "not found" not in str(exc):
            raise
        fallback_split = "validation" if split == "val" else split
        return load_dataset(
            repo_id, data_dir=config, split=fallback_split,
            revision=revision, streaming=True,
        )


def resolve_revision(repo_id: str, revision: str) -> str:
    try:
        from huggingface_hub import HfApi
        return str(HfApi().dataset_info(repo_id, revision=revision).sha)
    except Exception:
        return revision


def collect_ids(repo_id: str, config: str, split: str, revision: str, limit: int, seed: int) -> list[str]:
    candidates = ({"question_id": str(row["question_id"])} for row in load_stream(
        repo_id, config, split, revision,
    ))
    return [str(row["question_id"]) for row in stable_reservoir(candidates, limit, seed)]


def materialize_config(
    repo_id: str, config: str, split: str, revision: str,
    selected_ids: set[str], output_root: Path,
) -> dict[str, dict]:
    found: dict[str, dict] = {}
    image_root = output_root / "official_images" / config
    image_root.mkdir(parents=True, exist_ok=True)
    for row in load_stream(repo_id, config, split, revision):
        qid = str(row["question_id"])
        if qid not in selected_ids:
            continue
        if qid in found:
            raise ValueError(f"{config}: duplicate selected question_id {qid}")
        image_path = image_root / f"{qid}.jpg"
        row["image"].convert("RGB").save(image_path, format="JPEG", quality=95)
        copy = {key: json_safe(value) for key, value in row.items() if key != "image"}
        copy["image_path"] = str(image_path.resolve())
        copy["image_sha256"] = file_sha256(image_path)
        found[qid] = copy
        if len(found) == len(selected_ids):
            break
    missing = sorted(selected_ids - set(found))
    if missing:
        raise RuntimeError(f"{config}: {len(missing)} selected ids are missing; first={missing[:5]}")
    return found


def official_row(spec, source: dict, config: str, revision: str, target_letter: str) -> dict:
    image_path = source["image_path"]
    meta = source.get("meta", {})
    bbox = meta.get("rect") if isinstance(meta, dict) else None
    return {
        **spec.to_dict(),
        "dataset": "RIO-Bench-Obj-MC",
        "condition": RIO_CONDITION_BY_CONFIG[config],
        "image_path": image_path,
        "image_sha256": source["image_sha256"],
        "image_id": source.get("image_id"),
        "overlay_text": str(source.get("attack_word", "")),
        "bbox": bbox,
        "placement": "official-rio-bench",
        "overlay_area_fraction": None,
        "target_answer": target_letter,
        "target_content": str(source.get("attack_word", "")),
        "target_aliases": [target_letter, f"({target_letter})", str(source.get("attack_word", ""))],
        "choices": source["choices"],
        "attack_word": source.get("attack_word", ""),
        "rio_config": config,
        "rio_revision": revision,
        "official_attack_metadata": meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="turing-motors/RIO-Bench")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "render_manifest.jsonl"
    if manifest.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {manifest}")

    revision = resolve_revision(args.repo_id, args.revision)
    configs = RIO_CONFIG_GROUPS["obj_mc"]
    selected = collect_ids(args.repo_id, configs[0], args.split, revision, args.limit, args.seed)
    selected_ids = set(selected)
    indexed = {
        config: materialize_config(
            args.repo_id, config, args.split, revision, selected_ids, output_root,
        ) for config in configs
    }

    hard = indexed["obj_attack__mc_hard"]
    rows_written = 0
    for qid in selected:
        clean = indexed["obj_clean__mc_clean"][qid]
        hard_target, _ = target_letter_from_attack_word(hard[qid], args.seed)
        question_record = {
            "question_id": qid,
            "image": clean["image_path"],
            "text": clean["question"],
            "answer": clean["answer"],
            "choices": clean["choices"],
            "target_answer": hard_target,
            "attack_word": hard[qid].get("attack_word", ""),
            "task_type": "object",
            "category": "RIO-Bench Obj-MC",
        }
        spec = build_spec(question_record, output_root, args.seed)
        for condition in CONDITIONS:
            output = output_root / "images" / condition / f"{qid}.jpg"
            rendered = render_condition(spec, condition, output)
            append_jsonl(manifest, {
                **spec.to_dict(), **rendered,
                "dataset": "RIO-Bench-Obj-MC", "seed": args.seed,
                "choices": clean["choices"], "image_id": clean.get("image_id"),
                "rio_config": "derived-from-obj_clean__mc_clean",
                "rio_revision": revision,
            })
            rows_written += 1
        for config in configs[1:]:
            target, _ = target_letter_from_attack_word(indexed[config][qid], args.seed)
            append_jsonl(manifest, official_row(spec, indexed[config][qid], config, revision, target))
            rows_written += 1

    provenance = {
        "schema_version": "cta/rio-obj-mc-build-v1",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "resolved_revision": revision,
        "split": args.split,
        "data_dir_split_alias": {"val": "validation"},
        "configs": list(configs),
        "loader_policy": "named Hub config; exact top-level data_dir fallback when the client exposes only BuilderConfig default",
        "selection": "globally smallest SHA256(seed:question_id) on clean config",
        "seed": args.seed,
        "questions": len(selected),
        "conditions": list(CONDITIONS) + [RIO_CONDITION_BY_CONFIG[c] for c in configs[1:]],
        "rows": rows_written,
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "metric_boundary": "Final numbers require the official RIO evaluator; generated CTA conditions preserve the original Obj-MC question and hard-condition target.",
        "scene_tap_boundary": "The public HF card does not list SceneTAP configs; scene_coherent remains an in-house plaque and is not full SceneTAP.",
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()

