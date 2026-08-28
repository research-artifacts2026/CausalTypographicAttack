#!/usr/bin/env python3
"""Build a portable, image-first SCEI counterfactual twin dataset.

Each selected source produces three files:

* an exact clean-image copy;
* a scene-adaptive carrier with one mechanically false field; and
* an area-, geometry-, and layout-matched corrected control.

Selection and counterfactual families are frozen before the planner is loaded.
Per-item records make the long-running build safely resumable without silently
duplicating manifest rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import file_sha256
from cta.scei_attack import (
    REQUESTED_COUNTERFACTUAL_FAMILIES,
    compile_counterfactual,
    fallback_scene_plan,
    parse_scene_plan,
    planner_prompt,
    registered_evidence_text,
    render_carrier,
    validate_record,
    verification_question,
)
from cta.scei_batch import (
    assign_family_stratified_splits,
    freeze_selection,
    read_json_records,
    safe_item_slug,
)


DEFAULT_SCHEMA = "cta/scei-image-dataset-v1"
VARIANTS = ("clean", "attack_false", "control_true")


def _git_head() -> str:
    try:
        root = Path(__file__).resolve().parents[1]
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "not-a-git-checkout"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _copy_clean(source: Path, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    return {
        "image_path": str(output.resolve()),
        "image_relative_path": output.as_posix(),
        "image_sha256": file_sha256(output),
        "mask_path": None,
        "mask_relative_path": None,
        "mask_sha256": None,
        "carrier_quad": None,
        "overlay_area_fraction": 0.0,
        "renderer": "exact-source-copy-v1",
    }


def _relative_artifact(artifact: dict[str, Any], output_root: Path) -> dict[str, Any]:
    row = dict(artifact)
    row["image_relative_path"] = Path(row["image_path"]).resolve().relative_to(output_root).as_posix()
    if row.get("mask_path"):
        row["mask_relative_path"] = Path(row["mask_path"]).resolve().relative_to(output_root).as_posix()
    else:
        row["mask_relative_path"] = None
    return row


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _preview(records: list[dict[str, Any]], output_root: Path) -> Path:
    """Write one clean/false/true row per family for a compact visual audit."""
    representatives: dict[str, dict[str, Any]] = {}
    for record in records:
        representatives.setdefault(str(record["family"]), record)
    chosen = [representatives[family] for family in REQUESTED_COUNTERFACTUAL_FAMILIES if family in representatives]
    thumb_w, thumb_h = 320, 230
    header_h, label_h, gutter = 44, 30, 12
    canvas = Image.new(
        "RGB",
        (gutter + 3 * (thumb_w + gutter), header_h + len(chosen) * (label_h + thumb_h + gutter)),
        (244, 246, 249),
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(22, bold=True)
    label_font = _font(17, bold=True)
    small_font = _font(14)
    headings = ("CLEAN", "FALSE COUNTERFACTUAL", "CORRECTED TWIN")
    for column, heading in enumerate(headings):
        x = gutter + column * (thumb_w + gutter)
        draw.text((x + 6, 10), heading, fill=(24, 32, 48), font=title_font)
    for row_index, record in enumerate(chosen):
        y = header_h + row_index * (label_h + thumb_h + gutter)
        family = str(record["family"]).replace("_", " ").upper()
        draw.text((gutter + 6, y + 4), family, fill=(38, 68, 105), font=label_font)
        variants = {entry["variant"]: entry for entry in record["variants"]}
        for column, variant in enumerate(VARIANTS):
            x = gutter + column * (thumb_w + gutter)
            path = output_root / variants[variant]["image_relative_path"]
            with Image.open(path) as source:
                contained = ImageOps.contain(
                    source.convert("RGB"), (thumb_w, thumb_h), method=Image.Resampling.LANCZOS
                )
            tile = Image.new("RGB", (thumb_w, thumb_h), (234, 237, 242))
            tile.paste(contained, ((thumb_w - contained.width) // 2, (thumb_h - contained.height) // 2))
            canvas.paste(tile, (x, y + label_h))
            if column:
                draw.rectangle((x, y + label_h, x + thumb_w - 1, y + label_h + thumb_h - 1), outline=(77, 92, 117), width=2)
        draw.text(
            (gutter + 3 * (thumb_w + gutter) - 180, y + 7),
            str(record["target_label"]),
            fill=(68, 76, 92),
            font=small_font,
        )
    output = output_root / "preview_contact_sheet.jpg"
    canvas.save(output, quality=94, subsampling=0)
    return output


def _dataset_card(provenance: dict[str, Any]) -> str:
    family_lines = "\n".join(
        f"- `{family}`: {count}" for family, count in provenance["family_counts"].items()
    )
    split_lines = "\n".join(
        f"- `{split}`: {count}" for split, count in provenance.get("split_item_counts", {}).items()
    ) or "- not assigned"
    return f"""# {provenance['dataset_name']}

This image-first dataset contains {provenance['items']} source scenes and
{provenance['image_files']} portable images. Every source scene has an exact
clean copy, a scene-adaptive false counterfactual carrier, and a geometry- and
layout-matched corrected carrier.

## Variants

- `clean`: an exact copy of the registered source image.
- `attack_false`: one mechanically invalid field on a scene-adaptive carrier.
- `control_true`: the corresponding corrected field on the same carrier geometry.

## Counterfactual families

{family_lines}

## Splits

{split_lines}

Splits are stratified independently within every counterfactual family and are
assigned before planner or victim inference.  The false and corrected twins of
an item always remain in the same split.

## Symbolic truth generation

Generator: `{provenance['record_generator']}`.  Each item stores the changed
field, difficulty, solver expression, numeric parameters, printed record, and
independently recomputed residual.  False and corrected records differ in
exactly one pipe-delimited field.  A dataset row is not evidence of attack
success; victim evaluation is a separate stage.

## Important boundary

The carrier uses deterministic perspective, local color, texture, placement,
and shadow adaptation. It is synthetic compositing, not diffusion inpainting,
physical capture, or proof of attack success. Attack success must be measured
separately against a frozen victim protocol.

See `manifest.jsonl`, `selection.jsonl`, and `provenance.json` for registered
claims, hashes, masks, source paths, and generation metadata.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_manifest = Path(config["source_manifest"]).resolve()
    output_root = Path(config["output_root"]).resolve()
    expected_items = int(config["expected_items"])
    seed = int(config.get("seed", 20260827))
    offset = int(config.get("offset", 0))
    max_area = float(config.get("max_area_fraction", 0.15))
    attempts_allowed = int(config.get("max_planner_attempts", 3))
    require_valid = bool(config.get("require_valid_plans", False))
    schema = str(config.get("schema_version", DEFAULT_SCHEMA))
    dataset_name = str(config.get("dataset_name", f"SCEI-Images-{expected_items}"))
    record_generator = str(config.get("record_generator", "canonical_v1"))
    record_seed = int(config.get("record_seed", seed))
    families = tuple(config.get("counterfactual_families", REQUESTED_COUNTERFACTUAL_FAMILIES))
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}; use --resume")
    if attempts_allowed < 1 or attempts_allowed > 3:
        raise ValueError("max_planner_attempts must be between one and three")
    output_root.mkdir(parents=True, exist_ok=True)
    records_root = output_root / "records"
    records_root.mkdir(parents=True, exist_ok=True)

    selection_path = output_root / "selection.jsonl"
    if selection_path.is_file():
        selection = read_json_records(selection_path)
    else:
        selection = freeze_selection(
            source_manifest,
            seed=seed,
            offset=offset,
            limit=expected_items,
            families=families,
        )
        split_counts = config.get("split_counts_per_family")
        if split_counts:
            selection = assign_family_stratified_splits(
                selection,
                split_counts_per_family=split_counts,
                seed=int(config.get("split_seed", seed)),
            )
        _write_jsonl(selection_path, selection)
    if len(selection) != expected_items:
        raise ValueError(f"frozen selection has {len(selection)} rows, expected {expected_items}")

    completed: dict[str, dict[str, Any]] = {}
    for record_path in records_root.glob("*.json"):
        value = json.loads(record_path.read_text(encoding="utf-8"))
        completed[str(value["item_id"])] = value
    pending = [row for row in selection if row["item_id"] not in completed]
    planner_config = config.get("planner_model")
    planner = None
    if pending and planner_config:
        # Keep deterministic fallback-only dataset builds independent of the
        # heavyweight inference stack (torch/transformers).
        from cta.model import build_model_adapter

        planner = build_model_adapter(planner_config)
    semantic_signatures: set[str] = {
        str(value.get("semantic_signature"))
        for value in completed.values()
        if value.get("semantic_signature")
    }

    for sample in tqdm(pending, desc="SCEI image dataset"):
        item_id = str(sample["item_id"])
        slug = safe_item_slug(item_id)
        source = Path(sample["source_path"]).resolve()
        source_hash = file_sha256(source)
        if source_hash != sample["source_sha256"]:
            raise ValueError(f"{item_id}: source SHA-256 mismatch")
        label = re.sub(r"\s+", " ", str(sample["target_label"]).strip().lower())
        family = str(sample["family"])
        visible_labels = [str(value) for value in sample.get("visible_labels", [label])]
        variant_key = item_id
        if record_generator == "diverse_v2":
            for collision_index in range(100):
                candidate_key = item_id if collision_index == 0 else f"{item_id}:collision-{collision_index}"
                candidate = compile_counterfactual(
                    label,
                    family,
                    variant_key=candidate_key,
                    seed=record_seed,
                )
                candidate_signature = hashlib.sha256(
                    f"{family}\n{candidate.false_measurement}\n{candidate.true_measurement}".encode("utf-8")
                ).hexdigest()
                if candidate_signature not in semantic_signatures:
                    counterfactual = candidate
                    variant_key = candidate_key
                    semantic_signature = candidate_signature
                    semantic_signatures.add(candidate_signature)
                    break
            else:
                raise RuntimeError(f"{item_id}: unable to produce a unique semantic record")
        elif record_generator == "canonical_v1":
            counterfactual = compile_counterfactual(label, family)
            semantic_signature = hashlib.sha256(
                f"{family}\n{counterfactual.false_measurement}\n{counterfactual.true_measurement}".encode("utf-8")
            ).hexdigest()
        else:
            raise ValueError(f"unsupported record_generator: {record_generator!r}")
        validate_record(counterfactual)
        prompt = planner_prompt(label, visible_labels, counterfactual)
        raw_outputs: list[str] = []
        validation_errors: list[str] = []
        plan = None
        if planner is not None:
            for attempt in range(1, attempts_allowed + 1):
                retry_prompt = prompt
                if validation_errors:
                    retry_prompt += (
                        "\nThe previous response failed validation. Return a shorter JSON object that obeys every "
                        f"constraint. Validation issue: {validation_errors[-1]}"
                    )
                raw = str(planner.infer(str(source), retry_prompt))
                raw_outputs.append(raw)
                try:
                    plan = parse_scene_plan(raw, label)
                    break
                except ValueError as exc:
                    validation_errors.append(str(exc))
        else:
            validation_errors.append("planner disabled by configuration; deterministic fallback used")
        planner_valid = plan is not None
        if plan is None:
            plan = fallback_scene_plan(label, family, item_id)
            if require_valid:
                raise RuntimeError(f"{item_id}: planner failed after {attempts_allowed} attempts")

        clean_suffix = source.suffix.lower() if source.suffix.lower() in {".jpg", ".jpeg", ".png"} else ".jpg"
        clean_output = output_root / "images" / "clean" / f"{slug}{clean_suffix}"
        clean = _copy_clean(source, clean_output)
        clean["image_relative_path"] = clean_output.relative_to(output_root).as_posix()

        false_output = output_root / "images" / "attack_false" / f"{slug}.jpg"
        false_mask = output_root / "masks" / "attack_false" / f"{slug}.png"
        false = _relative_artifact(
            render_carrier(
                source, plan, counterfactual, "false", "scene", false_output, item_id,
                mask_output=false_mask, max_area_fraction=max_area,
            ).to_dict(),
            output_root,
        )
        true_output = output_root / "images" / "control_true" / f"{slug}.jpg"
        true_mask = output_root / "masks" / "control_true" / f"{slug}.png"
        true = _relative_artifact(
            render_carrier(
                source, plan, counterfactual, "true", "scene", true_output, item_id,
                mask_output=true_mask, max_area_fraction=max_area,
            ).to_dict(),
            output_root,
        )
        if false["carrier_quad"] != true["carrier_quad"]:
            raise RuntimeError(f"{item_id}: false/true carrier geometry differs")
        if false["mask_sha256"] != true["mask_sha256"]:
            raise RuntimeError(f"{item_id}: false/true carrier masks differ")

        common = {
            "schema_version": schema,
            "item_id": item_id,
            "selection_index": int(sample["selection_index"]),
            "split": str(sample.get("split", "unspecified")),
            "dataset": str(config.get("dataset", "COCO")),
            "source_path": str(source),
            "source_sha256": source_hash,
            "target_label": label,
            "source_target_label": str(sample.get("source_target_label", label)),
            "visible_labels": visible_labels,
            "family": family,
            "compatibility_score": int(sample.get("compatibility_score", 0)),
            "scenario_id": counterfactual.scenario_id,
            "record_variant_key": variant_key,
            "semantic_signature": semantic_signature,
            "plan": plan.to_dict(),
            "planner_valid": planner_valid,
            "planner_attempts": len(raw_outputs),
            "planner_raw_outputs": raw_outputs,
            "planner_validation_errors": validation_errors,
            "record": counterfactual.to_dict(),
        }
        variants = [
            {
                **clean,
                "variant": "clean",
                "truth": "source",
                "claim": None,
                "verification_question": None,
                "correct_answer": None,
                "registered_read_text": None,
            },
            {
                **false,
                "variant": "attack_false",
                "truth": "false",
                "claim": counterfactual.false_claim,
                "verification_question": verification_question(counterfactual, "false", "yesno", "semantic"),
                "correct_answer": "NO",
                "registered_read_text": registered_evidence_text(counterfactual, "false"),
            },
            {
                **true,
                "variant": "control_true",
                "truth": "true",
                "claim": counterfactual.true_claim,
                "verification_question": verification_question(counterfactual, "true", "yesno", "semantic"),
                "correct_answer": "YES",
                "registered_read_text": registered_evidence_text(counterfactual, "true"),
            },
        ]
        item_record = {**common, "variants": variants}
        _write_json(records_root / f"{slug}.json", item_record)
        completed[item_id] = item_record

    ordered_records = [completed[str(row["item_id"])] for row in selection]
    manifest_rows = []
    for item in ordered_records:
        for variant in item["variants"]:
            manifest_rows.append({
                **{key: value for key, value in item.items() if key not in {"variants", "planner_raw_outputs"}},
                **variant,
            })
    manifest_path = output_root / "manifest.jsonl"
    _write_jsonl(manifest_path, manifest_rows)
    preview_path = _preview(ordered_records, output_root)
    family_counts = dict(sorted(Counter(row["family"] for row in ordered_records).items()))
    split_counts = dict(sorted(Counter(str(row.get("split", "unspecified")) for row in ordered_records).items()))
    difficulty_counts = dict(sorted(Counter(str(row["record"].get("difficulty", "canonical")) for row in ordered_records).items()))
    scenario_counts = dict(sorted(Counter(str(row["scenario_id"]) for row in ordered_records).items()))
    provenance = {
        "schema_version": schema,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "hostname": platform.node(),
        "dataset": str(config.get("dataset", "COCO")),
        "dataset_name": dataset_name,
        "items": len(ordered_records),
        "variants": list(VARIANTS),
        "image_files": len(manifest_rows),
        "mask_files": 2 * len(ordered_records),
        "family_counts": family_counts,
        "split_item_counts": split_counts,
        "difficulty_counts": difficulty_counts,
        "scenario_counts": scenario_counts,
        "unique_semantic_signatures": len({row["semantic_signature"] for row in ordered_records}),
        "record_generator": record_generator,
        "record_seed": record_seed,
        "planner_valid_plans": sum(bool(row["planner_valid"]) for row in ordered_records),
        "planner_fallback_plans": sum(not bool(row["planner_valid"]) for row in ordered_records),
        "selection_seed": seed,
        "selection_offset": offset,
        "max_area_fraction": max_area,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": file_sha256(source_manifest),
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "selection": str(selection_path),
        "selection_sha256": file_sha256(selection_path),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "preview": str(preview_path),
        "preview_sha256": file_sha256(preview_path),
        "planner": planner.provenance() if planner is not None else planner_config,
        "planner_boundary": (
            "the planner sees the clean image, registered anchor label, visible labels, and invariant family; "
            "it never sees victim outputs and does not choose numeric truth values"
        ),
        "renderer_boundary": (
            "deterministic scene-adaptive perspective, tone, texture, placement, and shadow compositing; "
            "not diffusion inpainting or physical capture"
        ),
        "evaluation_boundary": "this artifact contains images and registered truth controls, not victim-model outcomes",
    }
    _write_json(output_root / "provenance.json", provenance)
    (output_root / "DATASET_CARD.md").write_text(_dataset_card(provenance), encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
