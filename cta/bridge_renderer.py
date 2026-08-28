"""Matched delivery-layer renderers for frozen Causal-Bridge text.

Both branches use the same registered title, proposition, bridge conclusion,
status line, canvas, bounding box, and font geometry.  The scene branch changes
only carrier delivery: a local tone palette, a small deterministic perspective
warp, and a soft shadow.  It is synthetic compositing, not camera capture.
"""

from __future__ import annotations

import hashlib
import io
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

from .question_bench import _ensure_canvas, file_sha256


def _font(size: int, bold: bool = False):
    names = [
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
        f"/usr/share/fonts/truetype/liberation2/LiberationSans{'-Bold' if bold else '-Regular'}.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def _palette(image: Image.Image, box: tuple[int, int, int, int], mode: str):
    if mode == "flat":
        return (248, 248, 246, 252), (24, 27, 31, 255), (32, 91, 151, 255)
    crop = image.crop(box).resize((1, 1), Image.Resampling.BOX)
    red, green, blue = crop.getpixel((0, 0))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    if luminance < 112:
        background = tuple(min(235, round(channel * 0.58 + 70)) for channel in (red, green, blue)) + (242,)
        foreground = (246, 246, 241, 255)
        accent = (244, 181, 73, 255)
    else:
        background = tuple(min(248, round(channel * 0.55 + 102)) for channel in (red, green, blue)) + (244,)
        foreground = (22, 26, 29, 255)
        accent = (126, 63, 37, 255)
    return background, foreground, accent


def _layout(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    title: str,
    proposition: str,
    conclusion: str,
    status: str,
):
    margin = max(10, round(width * 0.025))
    for size in range(max(18, round(height * 0.095)), 9, -1):
        body = _font(size, bold=True)
        title_font = _font(max(10, round(size * 0.72)), bold=True)
        status_font = _font(max(10, round(size * 0.76)), bold=True)
        chars = max(22, int((width - 2 * margin) / (size * 0.56)))
        proposition_lines = textwrap.wrap(proposition, chars) or [proposition]
        conclusion_lines = textwrap.wrap(conclusion, chars) or [conclusion]
        line_h = body.getbbox("Ag")[3] - body.getbbox("Ag")[1] + max(3, size // 5)
        title_h = title_font.getbbox("Ag")[3] - title_font.getbbox("Ag")[1] + 4
        status_h = status_font.getbbox("Ag")[3] - status_font.getbbox("Ag")[1] + 4
        divider_gap = max(8, size // 2)
        needed = (
            2 * margin + title_h + divider_gap +
            (len(proposition_lines) + len(conclusion_lines)) * line_h +
            status_h + 2 * divider_gap
        )
        if needed <= height:
            return title_font, body, status_font, proposition_lines, conclusion_lines, line_h, margin
    raise ValueError("registered Bridge text cannot fit the frozen panel geometry")


def _card(
    size: tuple[int, int],
    palette,
    title: str,
    proposition: str,
    conclusion: str,
    status: str,
) -> Image.Image:
    width, height = size
    background, foreground, accent = palette
    card = Image.new("RGBA", size, background)
    draw = ImageDraw.Draw(card)
    title_font, body, status_font, proposition_lines, conclusion_lines, line_h, margin = _layout(
        draw, width, height, title, proposition, conclusion, status
    )
    y = margin
    draw.text((margin, y), title, font=title_font, fill=accent)
    y += title_font.getbbox("Ag")[3] - title_font.getbbox("Ag")[1] + 6
    draw.line((margin, y, width - margin, y), fill=accent, width=max(2, width // 260))
    y += max(7, line_h // 2)
    for line in proposition_lines:
        draw.text((margin, y), line, font=body, fill=foreground)
        y += line_h
    y += max(3, line_h // 4)
    for line in conclusion_lines:
        draw.text((margin, y), line, font=body, fill=accent)
        y += line_h
    draw.text((margin, height - margin - status_font.getbbox("Ag")[3]), status, font=status_font, fill=accent)
    draw.rounded_rectangle(
        (1, 1, width - 2, height - 2), radius=max(6, height // 24),
        outline=accent, width=max(2, width // 220),
    )
    return card


def _quad(box: tuple[int, int, int, int], item_id: str):
    x0, y0, x1, y1 = box
    skew = max(3, round((x1 - x0) * 0.022))
    digest = hashlib.sha256(f"{item_id}:bridge-renderer-v1".encode()).digest()
    if digest[0] % 2:
        return [(x0 + skew, y0), (x1, y0 + skew), (x1 - skew, y1), (x0, y1 - skew)]
    return [(x0, y0 + skew), (x1 - skew, y0), (x1, y1 - skew), (x0 + skew, y1)]


def _perspective_coefficients(source, destination):
    matrix, vector = [], []
    for (x, y), (u, v) in zip(destination, source):
        matrix.extend(([x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]))
        vector.extend((u, v))
    return np.linalg.solve(np.asarray(matrix), np.asarray(vector))


def _save_jpeg(image: Image.Image, output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95, subsampling=0)
    output.write_bytes(buffer.getvalue())
    return file_sha256(output)


def render_bridge_carrier(
    source: str | Path,
    output: str | Path,
    item_id: str,
    bbox: tuple[int, int, int, int],
    mode: str,
    title: str,
    proposition: str,
    conclusion: str,
    status: str,
) -> dict:
    if mode not in {"flat", "scene"}:
        raise ValueError(f"unsupported renderer mode: {mode}")
    image = _ensure_canvas(str(source))
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
        raise ValueError(f"bbox {bbox} is outside canvas {image.size}")
    width, height = x1 - x0, y1 - y0
    card = _card((width, height), _palette(image, bbox, mode), title, proposition, conclusion, status)
    canvas = image.convert("RGBA")
    if mode == "flat":
        canvas.alpha_composite(card, (x0, y0))
        quad = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        renderer = "matched-flat-bridge-v1"
    else:
        quad = _quad(bbox, item_id)
        source_points = [(0, 0), (width, 0), (width, height), (0, height)]
        coefficients = _perspective_coefficients(source_points, quad)
        warped = card.transform(
            image.size, Image.Transform.PERSPECTIVE, coefficients,
            resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0, 0),
        )
        alpha = Image.new("L", card.size, 255).transform(
            image.size, Image.Transform.PERSPECTIVE, coefficients,
            resample=Image.Resampling.BICUBIC, fillcolor=0,
        )
        shadow_alpha = alpha.filter(ImageFilter.GaussianBlur(max(2, round(image.width * 0.004))))
        shifted = Image.new("L", image.size, 0)
        shifted.paste(shadow_alpha, (max(2, image.width // 180), max(2, image.height // 150)))
        shadow = Image.new("RGBA", image.size, (15, 15, 15, 0))
        shadow.putalpha(shifted.point(lambda value: round(value * 0.25)))
        canvas = Image.alpha_composite(canvas, shadow)
        canvas = Image.alpha_composite(canvas, warped)
        renderer = "matched-scene-bridge-v1"
    output = Path(output)
    return {
        "image_path": str(output.resolve()),
        "image_sha256": _save_jpeg(canvas, output),
        "bbox": list(bbox),
        "carrier_quad": [[round(float(x), 3), round(float(y), 3)] for x, y in quad],
        "overlay_area_fraction": width * height / (image.width * image.height),
        "renderer": renderer,
        "synthetic_scene_integration": mode == "scene",
    }
