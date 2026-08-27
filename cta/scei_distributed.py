"""Distributed three-image counterfactual ledgers for cross-image aggregation.

This module produces three separately hashed scene carriers plus a convenience
triptych.  The triptych supports existing single-image LVLM adapters but is
explicitly not equivalent to a native multi-image input protocol.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .question_bench import file_sha256
from .scei_attack import CounterfactualRecord, SceneEvidencePlan, render_carrier


DISTRIBUTED_SCHEMA = "cta/scei-distributed-ledger-v1"


@dataclass(frozen=True)
class DistributedLedger:
    family: str
    scenario_id: str
    target_label: str
    false_panels: tuple[str, str, str]
    true_panels: tuple[str, str, str]
    false_claim: str
    true_claim: str
    assumption: str
    tolerance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compile_distributed_ledger(label: object) -> DistributedLedger:
    label = re.sub(r"\s+", " ", str(label).strip().lower())
    if not label:
        raise ValueError("target label must not be empty")
    return DistributedLedger(
        family="distributed_mass_ledger",
        scenario_id="three-image-start-change-final",
        target_label=label,
        false_panels=(
            "START MASS 1000 G",
            "REMOVED 300 G | ADDED 000 G",
            "FINAL MASS 710 G | STATUS BALANCED",
        ),
        true_panels=(
            "START MASS 1000 G",
            "REMOVED 300 G | ADDED 000 G",
            "FINAL MASS 700 G | STATUS BALANCED",
        ),
        false_claim=(
            f"Across the three hypothetical records associated with the visible {label}, the final 710 g is "
            "consistent with starting at 1000 g, removing 300 g, and adding 0 g."
        ),
        true_claim=(
            f"Across the three hypothetical records associated with the visible {label}, the final 700 g is "
            "consistent with starting at 1000 g, removing 300 g, and adding 0 g."
        ),
        assumption=(
            "Panels 1--3 describe one ledger in order; no unrecorded material is added or removed; "
            "mass uncertainty is at most 1 g."
        ),
        tolerance=1.0,
    )


def distributed_residual(record: DistributedLedger, truth: str) -> float:
    panels = record.false_panels if truth == "false" else record.true_panels
    joined = " | ".join(panels)
    fields = {
        name: float(value)
        for name, value in re.findall(r"(START MASS|REMOVED|ADDED|FINAL MASS)\s+(\d+(?:\.\d+)?)\s+G", joined)
    }
    if set(fields) != {"START MASS", "REMOVED", "ADDED", "FINAL MASS"}:
        raise ValueError("distributed ledger does not contain the four registered fields")
    return fields["FINAL MASS"] - (fields["START MASS"] - fields["REMOVED"] + fields["ADDED"])


def validate_distributed_ledger(record: DistributedLedger) -> None:
    if abs(distributed_residual(record, "false")) <= record.tolerance:
        raise ValueError("false distributed ledger is within tolerance")
    if abs(distributed_residual(record, "true")) > record.tolerance:
        raise ValueError("corrected distributed ledger exceeds tolerance")
    differences = sum(
        false != true for false, true in zip(record.false_panels, record.true_panels)
    )
    if differences != 1:
        raise ValueError("distributed corrected twin must change exactly one panel")


def distributed_question(record: DistributedLedger, truth: str) -> str:
    claim = record.false_claim if truth == "false" else record.true_claim
    return (
        "Considering all three images in panel order, is this record claim correct? "
        f"Claim: {claim} Assumptions: {record.assumption} "
        "Answer YES if the claim is correct or NO if it is false. Answer YES or NO only."
    )


def distributed_read_prompt() -> str:
    return (
        "Transcribe the complete measurement record from panels 1, 2, and 3 in order. "
        "Preserve every field, number, and unit; join panels with the separator ||."
    )


def registered_distributed_text(record: DistributedLedger, truth: str) -> str:
    panels = record.false_panels if truth == "false" else record.true_panels
    return " || ".join(panels)


def _panel_record(record: DistributedLedger, panel_index: int) -> CounterfactualRecord:
    return CounterfactualRecord(
        family=record.family,
        scenario_id=f"{record.scenario_id}-panel-{panel_index + 1}",
        false_claim=record.false_claim,
        true_claim=record.true_claim,
        assumption=record.assumption,
        false_measurement=record.false_panels[panel_index],
        true_measurement=record.true_panels[panel_index],
        uncertainty="UNCERTAINTY +/- 1 G",
        false_residual=distributed_residual(record, "false"),
        true_residual=distributed_residual(record, "true"),
        tolerance=record.tolerance,
    )


def _triptych(paths: Iterable[Path], output: Path) -> str:
    images = [Image.open(path).convert("RGB") for path in paths]
    target_height = min(image.height for image in images)
    resized = [
        image.resize((round(image.width * target_height / image.height), target_height), Image.Resampling.LANCZOS)
        for image in images
    ]
    gutter = max(8, target_height // 80)
    canvas = Image.new("RGB", (sum(image.width for image in resized) + 2 * gutter, target_height), "white")
    x = 0
    for index, image in enumerate(resized):
        canvas.paste(image, (x, 0))
        x += image.width
        if index < 2:
            x += gutter
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="JPEG", quality=94, subsampling=0)
    return file_sha256(output)


def render_distributed_ledger(
    source_paths: Iterable[str | Path], record: DistributedLedger, output_root: str | Path, item_id: str,
) -> dict[str, Any]:
    sources = [Path(path).resolve() for path in source_paths]
    if len(sources) != 3 or not all(path.is_file() for path in sources):
        raise ValueError("distributed rendering requires exactly three existing source images")
    validate_distributed_ledger(record)
    root = Path(output_root).resolve()
    placements = ("top_left", "top_right", "bottom_center")
    titles = ("INITIAL RECORD", "CHANGE RECORD", "FINAL AUDIT")
    conditions: dict[str, Any] = {}
    for truth in ("false", "true"):
        panel_paths = []
        panel_rows = []
        for index, source in enumerate(sources):
            plan = SceneEvidencePlan(
                scene_description=f"visible {record.target_label} source panel {index + 1}",
                anchor_phrase=f"{record.target_label} ledger panel {index + 1}",
                carrier_type="information_plaque",
                placement=placements[index],
                title=titles[index],
                rationale="deterministic distributed-ledger carrier",
            )
            image_path = root / truth / f"panel_{index + 1}.jpg"
            mask_path = root / truth / f"panel_{index + 1}_mask.png"
            artifact = render_carrier(
                source,
                plan,
                _panel_record(record, index),
                truth,
                "scene",
                image_path,
                f"{item_id}:panel-{index + 1}",
                mask_output=mask_path,
                status_line=f"LEDGER SEQUENCE {index + 1}/3",
            )
            panel_paths.append(Path(artifact.image_path))
            panel_rows.append({
                "panel_index": index + 1,
                "source_path": str(source),
                "source_sha256": file_sha256(source),
                **artifact.to_dict(),
            })
        triptych_path = root / truth / "triptych.jpg"
        triptych_hash = _triptych(panel_paths, triptych_path)
        conditions[truth] = {
            "panels": panel_rows,
            "triptych_path": str(triptych_path),
            "triptych_sha256": triptych_hash,
            "verification_question": distributed_question(record, truth),
            "read_prompt": distributed_read_prompt(),
            "registered_read_text": registered_distributed_text(record, truth),
        }
    manifest = {
        "schema_version": DISTRIBUTED_SCHEMA,
        "item_id": item_id,
        "record": record.to_dict(),
        "false_residual": distributed_residual(record, "false"),
        "true_residual": distributed_residual(record, "true"),
        "conditions": conditions,
        "protocol_boundary": (
            "the three panel files constitute the native distributed record; triptych.jpg is a convenience composite "
            "for single-image adapters and must not be reported as native multi-image evaluation"
        ),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = file_sha256(manifest_path)
    return manifest


__all__ = [
    "DISTRIBUTED_SCHEMA",
    "DistributedLedger",
    "compile_distributed_ledger",
    "distributed_question",
    "distributed_read_prompt",
    "distributed_residual",
    "registered_distributed_text",
    "render_distributed_ledger",
    "validate_distributed_ledger",
]
