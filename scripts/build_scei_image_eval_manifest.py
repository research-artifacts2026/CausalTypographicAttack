#!/usr/bin/env python3
"""Build a frozen five-condition victim-evaluation manifest from SCEI images."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import file_sha256
from cta.scei_attack import (
    CONDITIONS,
    READ_CONDITIONS,
    CounterfactualRecord,
    SceneEvidencePlan,
    read_prompt,
    registered_evidence_text,
    render_carrier,
    semantic_token,
    verification_question,
)


ANSWER_CELLS = (("ab", "no_yes"), ("ab", "yes_no"), ("yesno", "semantic"))
VARIANTS = {"clean", "attack_false", "control_true"}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def git_head() -> str:
    root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "not-a-git-checkout"


def dataset_path(dataset_root: Path, row: dict, relative_key: str, absolute_key: str) -> Path | None:
    relative = row.get(relative_key)
    if relative:
        candidate = (dataset_root / relative).resolve()
        if candidate.is_file():
            return candidate
    absolute = row.get(absolute_key)
    return Path(absolute).resolve() if absolute else None


def rendered_fields(dataset_root: Path, row: dict) -> dict:
    image_path = dataset_path(dataset_root, row, "image_relative_path", "image_path")
    mask_path = dataset_path(dataset_root, row, "mask_relative_path", "mask_path")
    if image_path is None:
        raise ValueError(f"missing image path for {row.get('item_id')}/{row.get('variant')}")
    return {
        "image_path": str(image_path),
        "image_sha256": row["image_sha256"],
        "mask_path": str(mask_path) if mask_path else None,
        "mask_sha256": row.get("mask_sha256"),
        "carrier_quad": row.get("carrier_quad"),
        "overlay_area_fraction": float(row.get("overlay_area_fraction", 0.0)),
        "renderer": row["renderer"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-items", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--max-area-fraction", type=float, default=0.15)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    source_manifest = dataset_root / "manifest.jsonl"
    source_manifest_hash = file_sha256(source_manifest)
    rows = read_jsonl(source_manifest)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        item_id = str(row["item_id"])
        variant = str(row["variant"])
        if variant in grouped[item_id]:
            raise ValueError(f"duplicate variant: {item_id}/{variant}")
        grouped[item_id][variant] = row
    if any(set(values) != VARIANTS for values in grouped.values()):
        raise ValueError("every source item must contain clean, attack_false, and control_true")

    ordered = sorted(
        grouped.items(),
        key=lambda pair: (int(pair[1]["clean"]["selection_index"]), pair[0]),
    )
    if args.limit is not None:
        ordered = ordered[: args.limit]
    expected_items = args.limit if args.limit is not None else args.expected_items
    if len(ordered) != expected_items:
        raise ValueError(f"expected {expected_items} items, found {len(ordered)}")

    output_root = args.output_root.resolve()
    manifest_path = output_root / "render_manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {manifest_path}")
    output_root.mkdir(parents=True, exist_ok=True)

    rendered_rows = []
    for index, (item_id, variants) in enumerate(tqdm(ordered, desc="Build SCEI eval manifest")):
        clean = variants["clean"]
        scene_false = variants["attack_false"]
        scene_true = variants["control_true"]
        if scene_false["record"] != scene_true["record"] or scene_false["plan"] != scene_true["plan"]:
            raise ValueError(f"{item_id}: false/true record or plan mismatch")
        if scene_false.get("mask_sha256") != scene_true.get("mask_sha256"):
            raise ValueError(f"{item_id}: false/true carrier masks differ")
        if scene_false.get("carrier_quad") != scene_true.get("carrier_quad"):
            raise ValueError(f"{item_id}: false/true carrier geometry differs")

        record = CounterfactualRecord(**scene_false["record"])
        plan = SceneEvidencePlan(**scene_false["plan"])
        answer_format, option_order = ANSWER_CELLS[index % len(ANSWER_CELLS)]
        common = {
            "item_id": item_id,
            "selection_index": int(clean["selection_index"]),
            "dataset": str(clean.get("dataset", "COCO")),
            "stage": "scei-images-n300-fixed-victim-eval",
            "seed": args.seed,
            "source_path": str(Path(clean["source_path"]).resolve()),
            "source_sha256": clean["source_sha256"],
            "target_label": clean["target_label"],
            "visible_labels": clean.get("visible_labels", []),
            "family": record.family,
            "scenario_id": record.scenario_id,
            "answer_format": answer_format,
            "option_order": option_order,
            "counterbalance_cell": f"false:{answer_format}:{option_order}",
            "plan": plan.to_dict(),
            "record": record.to_dict(),
            "planner_valid": bool(scene_false.get("planner_valid", False)),
            "planner_attempts": int(scene_false.get("planner_attempts", 0)),
            "source_dataset_manifest": str(source_manifest),
            "source_dataset_manifest_sha256": source_manifest_hash,
        }

        for condition in CONDITIONS:
            truth = "true" if condition in {"clean_true", "scene_true"} else "false"
            correct_semantic = "yes" if truth == "true" else "no"
            target_semantic = "yes" if condition in {"flat_false", "scene_false"} else (
                "no" if truth == "true" else "yes"
            )
            if condition in {"clean_false", "clean_true"}:
                rendered = rendered_fields(dataset_root, clean)
            elif condition == "scene_false":
                rendered = rendered_fields(dataset_root, scene_false)
            elif condition == "scene_true":
                rendered = rendered_fields(dataset_root, scene_true)
            else:
                clean_image = dataset_path(dataset_root, clean, "image_relative_path", "image_path")
                if clean_image is None:
                    raise ValueError(f"missing clean image for {item_id}")
                artifact = render_carrier(
                    clean_image,
                    plan,
                    record,
                    "false",
                    "flat",
                    output_root / "images" / "flat_false" / f"{item_id}.jpg",
                    item_id,
                    mask_output=output_root / "masks" / "flat_false" / f"{item_id}.png",
                    max_area_fraction=args.max_area_fraction,
                )
                rendered = artifact.to_dict()

            measurement = record.false_measurement if truth == "false" else record.true_measurement
            rendered_rows.append({
                **common,
                **rendered,
                "condition": condition,
                "truth": truth,
                "verification_claim": record.false_claim if truth == "false" else record.true_claim,
                "verification_question": verification_question(record, truth, answer_format, option_order),
                "correct_semantic": correct_semantic,
                "target_semantic": target_semantic,
                "correct_answer_token": semantic_token(correct_semantic, answer_format, option_order),
                "target_answer_token": semantic_token(target_semantic, answer_format, option_order),
                "registered_read_text": registered_evidence_text(record, truth) if condition in READ_CONDITIONS else "NONE",
                "read_prompt": read_prompt(condition),
                "overlay_text": (
                    " | ".join((plan.title, plan.anchor_phrase, measurement, record.uncertainty, "STATUS: NOMINAL"))
                    if condition in READ_CONDITIONS else ""
                ),
                "requires_read": condition in READ_CONDITIONS,
            })

    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rendered_rows),
        encoding="utf-8",
    )
    provenance = {
        "schema_version": "cta/scei-image-eval-build-v1",
        "status": "frozen-before-victim-inference",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "builder_script": str(Path(__file__).resolve()),
        "builder_script_sha256": file_sha256(Path(__file__).resolve()),
        "hostname": platform.node(),
        "seed": args.seed,
        "items": len(ordered),
        "conditions": list(CONDITIONS),
        "rows": len(rendered_rows),
        "source_dataset_manifest": str(source_manifest),
        "source_dataset_manifest_sha256": source_manifest_hash,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "answer_cell_counts": dict(sorted(Counter(row["counterbalance_cell"] for row in rendered_rows[::len(CONDITIONS)]).items())),
        "family_counts": dict(sorted(Counter(row["family"] for row in rendered_rows[::len(CONDITIONS)]).items())),
        "max_area_fraction": args.max_area_fraction,
        "query_policy": "five answer rows and three independent read queries per item; no retries",
        "selection_boundary": "all claims, plans, pixels, questions, and answer cells frozen before victim inference",
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
