#!/usr/bin/env python3
"""Create a deterministic clean/AI-natural qualitative figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


INSET_BOXES = {
    "coco-000000134886-airplane": (120, 760, 650, 1080),
    "coco-000000475365-train": (160, 680, 650, 900),
    "coco-000000574810-cat": (780, 690, 1124, 1190),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path("assets/synthetic_natural_render/registry.json"))
    parser.add_argument("--output", type=Path, default=Path("assets/synthetic_natural_render/synthetic_natural_grid.png"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / args.registry).read_text(encoding="utf-8"))
    items = registry["items"]
    cell_w, image_h, gap, header_h, row_label_h = 560, 380, 24, 142, 42
    width = 2 * cell_w + 3 * gap
    height = header_h + len(items) * (image_h + row_label_h + gap) + gap
    canvas = Image.new("RGB", (width, height), "#f5f7fa")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 18), "AI-natural carrier feasibility pilot", font=font(31, True), fill="#14213d")
    draw.text((gap, 58), "Synthetic edits, not camera capture; n=3 pilot: 1/12 grounded model-item outcomes.",
              font=font(21), fill="#5a6472")
    x_positions = (gap, 2 * gap + cell_w)
    for col, title in enumerate(("Original source", "Synthetic natural-render")):
        draw.rounded_rectangle((x_positions[col], header_h - 38, x_positions[col] + cell_w, header_h - 4),
                               radius=9, fill="#dce5ef" if col == 0 else "#dff1e6")
        draw.text((x_positions[col] + 14, header_h - 33), title, font=font(21, True), fill="#14213d")
    for index, item in enumerate(items):
        y = header_h + index * (image_h + row_label_h + gap)
        source_raw = Image.open(root / item["source_path"])
        edited_raw = Image.open(root / item["image_path"])
        source = fit(source_raw, (cell_w, image_h))
        edited = fit(edited_raw, (cell_w, image_h))
        for col, image in enumerate((source, edited)):
            x = x_positions[col]
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + cell_w - 1, y + image_h - 1), outline="#aab4c0", width=2)
        crop = edited_raw.convert("RGB").crop(INSET_BOXES[item["item_id"]])
        crop.thumbnail((210, 125), Image.Resampling.LANCZOS)
        inset_x = x_positions[1] + cell_w - crop.width - 10
        inset_y = y + 10
        draw.rounded_rectangle(
            (inset_x - 5, inset_y - 5, inset_x + crop.width + 5, inset_y + crop.height + 5),
            radius=6, fill="white", outline="#26364d", width=3,
        )
        canvas.paste(crop, (inset_x, inset_y))
        label = f"{chr(97 + index)}) {item['target_label'].title()} — {item['registered_read_text']}"
        draw.text((gap, y + image_h + 7), label, font=font(20, True), fill="#273142")
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    print(output)


if __name__ == "__main__":
    main()
