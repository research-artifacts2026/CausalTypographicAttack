#!/usr/bin/env python3
"""Render and finalize a matched ContraLedger replay through SceneTAP.

The public SceneTAP SoM masks and TextDiffuser renderer are used unchanged.
Placement is supplied by ``plan_scenetap_local_qwen.py``.  One region is
planned from the false record and reused for its corrected twin.  Independent
shards write per-item records; ``--finalize-only`` assembles and audits the
immutable three-state manifest after every shard finishes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.contraledger_threeway import CONDITIONS
from cta.question_bench import file_sha256
from scripts.analyze_contraledger_threeway import audit_manifest


_RENDER_FIELDS = {
    "carrier_quad",
    "image_path",
    "image_sha256",
    "mask_path",
    "mask_sha256",
    "overlay_area_fraction",
    "renderer",
}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def strip_render_fields(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in _RENDER_FIELDS}


def stable_seed(item_id: str, truth: str, seed: int) -> int:
    """Return one paired diffusion seed for both semantic twins.

    ``truth`` remains in the public call signature so existing callers and
    provenance readers do not break, but it is deliberately excluded from the
    hash.  Using different random draws for the true and false record would
    confound the registered one-field intervention with stochastic texture and
    lighting changes.
    """
    digest = hashlib.sha256(
        f"contraledger-scenetap-v1:{seed}:{item_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def set_seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    import torch

    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def native_index(path: Path) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    rows = read_jsonl(path)
    audit_manifest(rows)
    indexed = {(str(row["item_id"]), str(row["condition"])): row for row in rows}
    return rows, indexed


def make_mask(size: tuple[int, int], bbox: tuple[int, int, int, int], output: Path) -> dict:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle((bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1), fill=255)
    output.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output)
    area = int(np.count_nonzero(np.asarray(mask))) / float(size[0] * size[1])
    return {"mask_path": str(output.resolve()), "mask_sha256": file_sha256(output), "area": area}


def resolve_bbox(row: dict, image: Image.Image, largest_inscribed_rectangle, find_text_region) -> tuple[tuple[int, int, int, int], dict]:
    masks = np.load(row["mask_path"], allow_pickle=True)
    region_index = int(row["plan"]["text_position_number"]) - 1
    if not 0 <= region_index < len(masks):
        raise ValueError(f"{row['question_id']}: planner region is out of range")
    target_mask = masks[region_index]["segmentation"]
    x, y, width, height = largest_inscribed_rectangle(target_mask, True)
    resolution = {"requested_region": region_index + 1, "rectangle_fallback": False}
    if width <= 1 or height <= 1:
        ys, xs = np.where(np.asarray(target_mask) == True)  # noqa: E712 - mask label is literally True
        if len(xs) == 0:
            raise ValueError(f"{row['question_id']}: selected SoM mask is empty")
        x, y = int(xs.min()), int(ys.min())
        width, height = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        resolution["rectangle_fallback"] = True
        resolution["fallback_reason"] = "public largest-inscribed rectangle was degenerate"
    mask_height, mask_width = np.asarray(target_mask).shape[:2]
    left = x / mask_width * image.width
    top = y / mask_height * image.height
    right = (x + width) / mask_width * image.width
    bottom = (y + height) / mask_height * image.height
    bbox = find_text_region(
        str(row["target_text"]), left, top, right, bottom,
        font_path="./fonts/arial.ttf", font_size=20, aspect_ratio_threshold=0.1,
    )
    l, t, r, b = [int(v) for v in bbox]
    l = max(0, min(image.width - 2, l)); r = max(l + 2, min(image.width, r))
    t = max(0, min(image.height - 2, t)); b = max(t + 2, min(image.height, b))
    if r - l < 8 or b - t < 8:
        # Keep the public region while avoiding a zero-area TextDiffuser input.
        l, t, r, b = [int(round(v)) for v in (left, top, right, bottom)]
        l = max(0, min(image.width - 8, l)); r = max(l + 8, min(image.width, r))
        t = max(0, min(image.height - 8, t)); b = max(t + 8, min(image.height, b))
        resolution["text_fit_fallback"] = True
        resolution["text_fit_reason"] = "public aspect-fit box was smaller than 8 pixels"
    else:
        resolution["text_fit_fallback"] = False
    return (l, t, r, b), resolution


def render_shard(args: argparse.Namespace) -> None:
    plans_path = args.plans.resolve()
    native_path = args.native_manifest.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_root = output / "records"
    records_root.mkdir(parents=True, exist_ok=True)
    _, native = native_index(native_path)
    plans = sorted(read_jsonl(plans_path), key=lambda row: str(row["question_id"]))
    selected = [row for index, row in enumerate(plans) if index % args.shard_count == args.shard_index]
    if not selected:
        raise ValueError("empty render shard")

    scenetap_root = args.scenetap_root.resolve()
    os.chdir(scenetap_root)
    sys.path.insert(0, str(scenetap_root))
    from utils.get_rectangle_by_mask import largest_inscribed_rectangle
    from utils.text_diffuser import TextDiffuser
    from utils.typo_attack_planner import find_text_region

    diffuser = TextDiffuser()
    completed = 0
    for plan in selected:
        item_id = str(plan["question_id"])
        record_path = records_root / f"{item_id}.json"
        if record_path.exists():
            completed += 1
            continue
        source = native[(item_id, "source_absent")]
        image = Image.open(plan["original_image_path"]).convert("RGB")
        if file_sha256(Path(plan["original_image_path"])) != str(source["source_sha256"]):
            raise ValueError(f"{item_id}: staged image is not the frozen source")
        bbox, bbox_resolution = resolve_bbox(
            plan, image, largest_inscribed_rectangle, find_text_region
        )
        mask_info = make_mask(
            image.size, bbox, output / "masks" / f"{item_id}.png"
        )
        pair = []
        planned_false_caption = str(plan.get("plan", {}).get("short_caption", "")).strip()
        false_text = str(native[(item_id, "record_false")]["registered_read_text"])
        for condition, truth in (("record_true", "true"), ("record_false", "false")):
            frozen = native[(item_id, condition)]
            text = str(frozen["registered_read_text"])
            if false_text.casefold() in planned_false_caption.casefold():
                # resolve_caption in the planner guarantees an exact attack
                # string under normal operation. Preserve the planner's scene
                # description and replace only the registered twin text.
                start = planned_false_caption.casefold().index(false_text.casefold())
                caption = (
                    planned_false_caption[:start]
                    + text
                    + planned_false_caption[start + len(false_text):]
                )
                caption_source = "local-qwen-scene-caption"
            else:
                caption = f"A natural printed measurement record in the scene reads '{text}'."
                caption_source = "fixed-fallback-caption"
            seed_value = stable_seed(item_id, truth, args.seed)
            set_seed(seed_value)
            started = time.time()
            result = diffuser.generate(
                [(bbox[0], bbox[1]), (bbox[2], bbox[3])],
                plan["original_image_path"],
                text,
                caption,
                radio="Two Points",
                scale_factor=2,
                regional_diffusion=True,
            )
            candidates = [candidate.resize(image.size) for candidate in result[0]]
            if not candidates:
                raise RuntimeError(f"{item_id}/{truth}: TextDiffuser returned no candidates")
            selected_path = output / "images" / condition / f"{item_id}.jpg"
            selected_path.parent.mkdir(parents=True, exist_ok=True)
            candidates[0].convert("RGB").save(selected_path, format="JPEG", quality=95)
            candidate_root = output / "candidates" / item_id / condition
            candidate_root.mkdir(parents=True, exist_ok=True)
            for index, candidate in enumerate(candidates):
                candidate.convert("RGB").save(
                    candidate_root / f"{index}.jpg", format="JPEG", quality=95
                )
            pair.append({
                **strip_render_fields(frozen),
                "schema_version": "cta/contraledger-threeway-delivery-item-v1",
                "delivery_method": "scenetap-public-components-local-qwen-planner",
                "image_path": str(selected_path.resolve()),
                "image_sha256": file_sha256(selected_path),
                "mask_path": mask_info["mask_path"],
                "mask_sha256": mask_info["mask_sha256"],
                "carrier_quad": [
                    [bbox[0], bbox[1]], [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]], [bbox[0], bbox[3]],
                ],
                "overlay_area_fraction": mask_info["area"],
                "renderer": "SceneTAP public TextDiffuser component",
                "scenetap_bbox": list(bbox),
                "scenetap_bbox_resolution": bbox_resolution,
                "scenetap_candidate_count": len(candidates),
                "scenetap_selected_candidate_index": 0,
                "scenetap_seed": seed_value,
                "scenetap_caption": caption,
                "scenetap_caption_source": caption_source,
                "scenetap_render_latency_s": round(time.time() - started, 4),
                "scenetap_planner_model": plan.get("planner_model"),
                "scenetap_plan": plan.get("plan"),
            })
        payload = {
            "item_id": item_id,
            "source": {
                **strip_render_fields(source),
                "schema_version": "cta/contraledger-threeway-delivery-item-v1",
                "delivery_method": "source-unmodified",
                "image_path": source["source_path"],
                "image_sha256": source["source_sha256"],
            },
            "record_true": pair[0],
            "record_false": pair[1],
        }
        temporary = record_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(record_path)
        completed += 1

    shard_provenance = {
        "schema_version": "cta/contraledger-scenetap-render-shard-v1",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "items": len(selected),
        "completed_items": completed,
        "plans": str(plans_path),
        "plans_sha256": file_sha256(plans_path),
        "native_manifest": str(native_path),
        "native_manifest_sha256": file_sha256(native_path),
        "candidate_selection": "candidate index zero, fixed before victim inference",
        "seed": args.seed,
        "victim_outputs_used": False,
    }
    (output / f"shard_{args.shard_index:02d}_of_{args.shard_count:02d}.json").write_text(
        json.dumps(shard_provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(shard_provenance, indent=2))


def finalize(args: argparse.Namespace) -> None:
    plans_path = args.plans.resolve()
    native_path = args.native_manifest.resolve()
    output = args.output_root.resolve()
    native_rows, native = native_index(native_path)
    item_ids = sorted({str(row["item_id"]) for row in native_rows})
    rows = []
    bbox_fallbacks = 0
    text_fit_fallbacks = 0
    render_seconds = 0.0
    for item_id in item_ids:
        record_path = output / "records" / f"{item_id}.json"
        if not record_path.is_file():
            raise FileNotFoundError(f"missing SceneTAP record: {record_path}")
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        if str(payload.get("item_id")) != item_id:
            raise ValueError(f"{item_id}: record id mismatch")
        triplet = [payload["source"], payload["record_true"], payload["record_false"]]
        if [row["condition"] for row in triplet] != list(CONDITIONS):
            raise ValueError(f"{item_id}: invalid condition ordering")
        for row in triplet[1:]:
            resolution = row.get("scenetap_bbox_resolution", {})
            bbox_fallbacks += int(bool(resolution.get("rectangle_fallback")))
            text_fit_fallbacks += int(bool(resolution.get("text_fit_fallback")))
            render_seconds += float(row.get("scenetap_render_latency_s", 0.0))
        rows.extend(triplet)
    audit = audit_manifest(rows)
    manifest = output / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    provenance = {
        "schema_version": "cta/contraledger-scenetap-render-v1",
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "delivery_method": "scenetap-public-components-local-qwen-planner",
        "items": len(item_ids),
        "rows": len(rows),
        "conditions": list(CONDITIONS),
        "family_counts": dict(sorted(Counter(
            native[(item_id, "record_false")]["family"] for item_id in item_ids
        ).items())),
        "plans": str(plans_path),
        "plans_sha256": file_sha256(plans_path),
        "native_manifest": str(native_path),
        "native_manifest_sha256": file_sha256(native_path),
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "manifest_audit": audit,
        "render_seconds_total": round(render_seconds, 4),
        "rectangle_fallback_rows": bbox_fallbacks,
        "text_fit_fallback_rows": text_fit_fallbacks,
        "matched_fields": [
            "source image", "question", "option map", "symbolic record",
            "registered read text", "semantic answers", "family", "item selection",
        ],
        "changed_factor": "delivery renderer and placement planner only",
        "candidate_selection": "TextDiffuser candidate index zero; no outcome filtering",
        "victim_outputs_used": False,
        "official_equivalence": False,
        "claim_boundary": (
            "Uses public SceneTAP SoM and TextDiffuser components with a local "
            "Qwen2.5-VL planner. It is not an exact replay of the unavailable "
            "official GPT-4o planner and does not use SceneTAP's original short "
            "target-token content."
        ),
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenetap-root", type=Path, default=Path("/disk2/fangxinyue/scenetap"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.finalize_only:
        finalize(args)
        return
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    render_shard(args)


if __name__ == "__main__":
    main()
