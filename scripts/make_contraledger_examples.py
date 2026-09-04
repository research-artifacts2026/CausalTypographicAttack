#!/usr/bin/env python3
"""Make outcome-oblivious ContraLedger method and family visualizations."""

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


FAMILY_LABELS = {
    "capacity_conservation": "Capacity / conservation",
    "causal_order": "Causal order",
    "geometry_feasibility": "Geometric feasibility",
    "phase_state": "Thermodynamic phase",
    "probability_ledger": "Probability ledger",
    "range_threshold": "Range / threshold",
    "temporal_ledger": "Temporal ledger",
    "unit_conversion": "Unit conversion",
}
CONDITION_LABELS = {
    "values_only_false": "False fields + neutral ID",
    "authority_false": "Same fields + authority",
    "explicit_conclusion_false": "Same fields + verdict",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def fit(image: Image.Image, size: tuple[int, int], color: str = "#E2E8F0") -> Image.Image:
    return ImageOps.pad(image.convert("RGB"), size, method=Image.Resampling.LANCZOS, color=color)


def carrier_crop(image: Image.Image, quad: list[list[float]], pad: float = 0.08) -> Image.Image:
    xs = [float(point[0]) for point in quad]
    ys = [float(point[1]) for point in quad]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    extra_x = max(18.0, (right - left) * pad)
    extra_y = max(18.0, (bottom - top) * pad)
    box = (
        max(0, int(left - extra_x)),
        max(0, int(top - extra_y)),
        min(image.width, int(right + extra_x)),
        min(image.height, int(bottom + extra_y)),
    )
    return image.crop(box)


def select_family_items(rows: list[dict]) -> dict[str, str]:
    candidates: dict[str, set[str]] = {family: set() for family in FAMILY_LABELS}
    for row in rows:
        family = str(row["family"])
        if family in candidates:
            candidates[family].add(str(row["item_id"]))
    selected = {}
    for family, item_ids in candidates.items():
        if not item_ids:
            raise ValueError(f"manifest has no items for {family}")
        selected[family] = min(
            item_ids,
            key=lambda item_id: hashlib.sha256(
                f"contraledger-figure-v1:{family}:{item_id}".encode()
            ).hexdigest(),
        )
    return selected


def build_method_figure(by_key: dict, item_id: str, output: Path) -> list[dict]:
    source_row = by_key[(item_id, "values_only_false")]
    canvas = Image.new("RGB", (3200, 1600), "#F8FAFC")
    draw = ImageDraw.Draw(canvas)
    draw.text((80, 52), "ContraLedger: isolate what makes a false record persuasive", font=font(54, True), fill="#0F172A")
    draw.text(
        (82, 120),
        "Same source, question, false measurement fields, carrier geometry, and mask; only the final cue changes.",
        font=font(28),
        fill="#475569",
    )
    panels = [
        ("Source scene", Path(source_row["source_path"]), None, "#334155"),
        (CONDITION_LABELS["values_only_false"], Path(by_key[(item_id, "values_only_false")]["image_path"]), by_key[(item_id, "values_only_false")], "#2563EB"),
        (CONDITION_LABELS["authority_false"], Path(by_key[(item_id, "authority_false")]["image_path"]), by_key[(item_id, "authority_false")], "#D97706"),
        (CONDITION_LABELS["explicit_conclusion_false"], Path(by_key[(item_id, "explicit_conclusion_false")]["image_path"]), by_key[(item_id, "explicit_conclusion_false")], "#DC2626"),
    ]
    source_path = Path(source_row["source_path"])
    used = [{
        "condition": "source",
        "path": str(source_path.resolve()),
        "sha256": file_sha256(source_path),
    }]
    for index, (label, path, row, color) in enumerate(panels):
        x = 70 + index * 785
        draw.rounded_rectangle((x, 190, x + 735, 1355), radius=24, fill="white", outline="#CBD5E1", width=3)
        draw.rounded_rectangle((x + 18, 210, x + 717, 284), radius=18, fill=color)
        draw.text((x + 42, 228), f"{chr(65 + index)}  {label}", font=font(28, True), fill="white")
        image = Image.open(path).convert("RGB")
        canvas.paste(fit(image, (680, 700), "white"), (x + 28, 310))
        if row is None:
            draw.text((x + 48, 1060), "Visible object and scene\nare fixed before attack.", font=font(29, True), fill="#334155", spacing=12)
        else:
            crop = carrier_crop(image, row["carrier_quad"])
            crop = fit(crop, (680, 250), "#E2E8F0")
            canvas.paste(crop, (x + 28, 1045))
            draw.rectangle((x + 28, 1045, x + 708, 1295), outline=color, width=5)
            used.append({
                "condition": row["condition"],
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "status_line": row["status_line"],
            })
    question = str(source_row["question"]).split(" Options:", 1)[0]
    draw.rounded_rectangle((80, 1382, 3120, 1568), radius=18, fill="#E0F2FE")
    footer = f"Frozen question ({source_row['question_polarity']}): {question}"
    draw.multiline_text(
        (112, 1403),
        "\n".join(textwrap.wrap(footer, width=185)),
        font=font(25, True),
        fill="#0C4A6E",
        spacing=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92, subsampling=0, optimize=True)
    return used


def build_family_gallery(by_key: dict, selected: dict[str, str], output: Path) -> list[dict]:
    canvas = Image.new("RGB", (3200, 2240), "#F8FAFC")
    draw = ImageDraw.Draw(canvas)
    draw.text((80, 50), "Eight scene-grounded, mechanically checkable counterfactual families", font=font(52, True), fill="#0F172A")
    draw.text((82, 118), "Examples are selected by a frozen item-ID hash, without model outputs.", font=font(28), fill="#475569")
    used = []
    for index, (family, label) in enumerate(FAMILY_LABELS.items()):
        row = by_key[(selected[family], "values_only_false")]
        image_path = Path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        col, row_index = index % 4, index // 4
        x, y = 70 + col * 785, 185 + row_index * 1000
        draw.rounded_rectangle((x, y, x + 735, y + 930), radius=22, fill="white", outline="#CBD5E1", width=3)
        draw.text((x + 28, y + 26), label, font=font(30, True), fill="#0F172A")
        draw.text((x + 28, y + 70), f"{row['target_label']}  |  {row['question_polarity']} question", font=font(22), fill="#64748B")
        canvas.paste(fit(image, (680, 560), "white"), (x + 28, y + 115))
        crop = fit(carrier_crop(image, row["carrier_quad"]), (680, 210), "#E2E8F0")
        canvas.paste(crop, (x + 28, y + 695))
        draw.rectangle((x + 28, y + 695, x + 708, y + 905), outline="#2563EB", width=4)
        used.append({
            "family": family,
            "item_id": row["item_id"],
            "path": str(image_path.resolve()),
            "sha256": file_sha256(image_path),
            "source_path": str(Path(row["source_path"]).resolve()),
            "source_sha256": str(row["source_sha256"]),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92, subsampling=0, optimize=True)
    return used


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    rows = read_jsonl(manifest)
    by_key = {(str(row["item_id"]), str(row["condition"])): row for row in rows}
    selected = select_family_items(rows)
    method_item = selected["unit_conversion"]
    output_dir = args.output_dir.resolve()
    method_path = output_dir / "contraledger_method.jpg"
    gallery_path = output_dir / "contraledger_family_gallery.jpg"
    method_inputs = build_method_figure(by_key, method_item, method_path)
    gallery_inputs = build_family_gallery(by_key, selected, gallery_path)
    sidecar = {
        "schema_version": "cta/contraledger-figure-v1",
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": file_sha256(Path(__file__).resolve()),
        "selection_policy": "minimum SHA-256 of contraledger-figure-v1:family:item_id; no victim outputs",
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "method_item": method_item,
        "family_items": selected,
        "inputs": {"method": method_inputs, "gallery": gallery_inputs},
        "outputs": {
            "method": {"path": str(method_path), "sha256": file_sha256(method_path)},
            "gallery": {"path": str(gallery_path), "sha256": file_sha256(gallery_path)},
        },
        "claim_boundary": "Qualitative frozen examples only; selection and generation do not use victim outcomes.",
    }
    (output_dir / "provenance.json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
