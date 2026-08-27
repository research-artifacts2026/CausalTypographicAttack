#!/usr/bin/env python3
"""Export a consolidated SCEI-Search dataset from frozen audited run evidence."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.question_bench import file_sha256
from cta.scei_adaptive import AdaptiveSceneDesign
from cta.scei_attack import (
    compile_counterfactual,
    recompute_record_residual,
    registered_evidence_text,
    render_carrier,
    validate_record,
    verification_question,
)
from cta.scei_batch import load_jsonl, safe_item_slug


def git_head() -> str:
    try:
        root = Path(__file__).resolve().parents[1]
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "not-a-git-checkout"


def copy_verified(source: Path, destination: Path, expected_hash: str) -> str:
    if not source.is_file() or file_sha256(source) != expected_hash:
        raise RuntimeError(f"source evidence hash mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied_hash = file_sha256(destination)
    if copied_hash != expected_hash:
        raise RuntimeError(f"copied evidence hash mismatch: {destination}")
    return copied_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite frozen dataset export: {output_root}")
    selection_path = run_root / "selection_manifest.json"
    results_path = run_root / "sample_results.jsonl"
    analysis_path = run_root / "analysis.json"
    lock_path = run_root / "run_lock.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    results = load_jsonl(results_path)
    by_id = {str(row["item_id"]): row for row in results}
    if len(selection) != len(results) or len(by_id) != len(results):
        raise RuntimeError("dataset export requires one unique terminal result per selected item")
    output_root.mkdir(parents=True)
    exported = []
    for item in selection:
        item_id = str(item["item_id"])
        result = by_id[item_id]
        events = load_jsonl(Path(result["events_path"]))
        attacks = sorted(
            (row for row in events if row.get("stage") == "attack"),
            key=lambda row: int(row["round"]),
        )
        if not attacks:
            raise RuntimeError(f"{item_id}: no attack image exists")
        record = compile_counterfactual(item["target_label"], item["family"])
        validate_record(record)
        item_root = output_root / "items" / safe_item_slug(item_id)
        clean_path = item_root / "clean.jpg"
        clean_hash = copy_verified(
            Path(item["source_path"]), clean_path, str(item["source_sha256"])
        )
        attack_rows = []
        for event in attacks:
            round_index = int(event["round"])
            image_destination = item_root / f"attack_round_{round_index:02d}.jpg"
            mask_destination = item_root / f"attack_round_{round_index:02d}_mask.png"
            image_hash = copy_verified(
                Path(event["image_path"]), image_destination, str(event["image_sha256"])
            )
            mask_hash = copy_verified(
                Path(event["mask_path"]), mask_destination, str(event["mask_sha256"])
            )
            attack_rows.append({
                "round": round_index,
                "image_path": str(image_destination),
                "image_sha256": image_hash,
                "mask_path": str(mask_destination),
                "mask_sha256": mask_hash,
                "design": event["design"],
                "carrier_quad": event["render"]["carrier_quad"],
                "overlay_area_fraction": event["render"]["overlay_area_fraction"],
                "answer_raw": event["answer_raw"],
                "parsed_semantic": event["parsed_semantic"],
                "read_raw": event["read_raw"],
                "exact_read_match": event["exact_read_match"],
                "feedback_class": event["feedback_class"],
                "strict_success": event["success"],
            })

        first = attacks[0]
        design = AdaptiveSceneDesign(**first["design"])
        twin_path = item_root / "corrected_twin_round_01.jpg"
        twin_mask_path = item_root / "corrected_twin_round_01_mask.png"
        twin = render_carrier(
            clean_path,
            design.scene_plan(),
            record,
            "true",
            "scene",
            twin_path,
            "adaptive-round-1",
            mask_output=twin_mask_path,
            status_line=design.status_line,
        )
        if twin.mask_sha256 != first["mask_sha256"]:
            raise RuntimeError(f"{item_id}: corrected twin mask differs from first-round false carrier")
        if [list(point) for point in twin.carrier_quad] != first["render"]["carrier_quad"]:
            raise RuntimeError(f"{item_id}: corrected twin geometry differs from first-round false carrier")
        exported.append({
            "schema_version": "cta/scei-search-dataset-item-v1",
            "item_id": item_id,
            "selection_index": item["selection_index"],
            "family": item["family"],
            "source_target_label": item["source_target_label"],
            "anchor_label": item["target_label"],
            "visible_labels": item["visible_labels"],
            "compatibility_score": item["compatibility_score"],
            "clean_path": str(clean_path),
            "clean_sha256": clean_hash,
            "record": record.to_dict(),
            "mechanical_false_residual": recompute_record_residual(record, "false"),
            "mechanical_true_residual": recompute_record_residual(record, "true"),
            "false_question": verification_question(record, "false", "yesno", "semantic"),
            "true_question": verification_question(record, "true", "yesno", "semantic"),
            "registered_false_read_text": registered_evidence_text(record, "false"),
            "registered_true_read_text": registered_evidence_text(record, "true"),
            "attack_candidates": attack_rows,
            "corrected_twin": twin.to_dict(),
            "terminal_outcome": result["terminal_outcome"],
            "success_at_1": result["success_at_1"],
            "success_at_2": result["success_at_2"],
        })

    manifest_path = output_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(exported, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": "cta/scei-search-dataset-export-v1",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "hostname": platform.node(),
        "source_run_root": str(run_root),
        "source_run_lock_sha256": file_sha256(lock_path),
        "source_selection_sha256": file_sha256(selection_path),
        "source_results_sha256": file_sha256(results_path),
        "source_analysis_sha256": file_sha256(analysis_path),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "items": len(exported),
        "families": dict(sorted(Counter(row["family"] for row in exported).items())),
        "images": {
            "clean": len(exported),
            "false_attack_candidates": sum(len(row["attack_candidates"]) for row in exported),
            "corrected_twins": len(exported),
        },
        "renderer_boundary": (
            "deterministic scene-adaptive perspective carriers; synthetic composites, not AI inpainting and not "
            "camera-captured physical attacks"
        ),
        "selection_boundary": (
            "all 50 selected items are exported, including every attack failure; outcomes never filter the dataset"
        ),
    }
    provenance_path = output_root / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "DATASET_CARD.md").write_text(
        "# SCEI-Search counterfactual suite\n\n"
        "This export contains all 50 frozen COCO items, every bounded false-record candidate, and an area/layout-"
        "matched corrected twin for the first candidate. The eight single-image families are range/threshold, unit "
        "conversion, temporal ledger, capacity conservation, causal order, geometry feasibility, probability ledger, "
        "and phase/state.\n\n"
        "Strict success requires a correct clean answer, acceptance of the false record, and exact independent "
        "transcription. Do not treat answer-only flips as grounded successes. These are deterministic synthetic scene "
        "carriers, not camera captures or AI-inpainted images. See provenance.json for source hashes and scope.\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
