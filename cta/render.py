from __future__ import annotations

import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .generation import AttackText


@dataclass(frozen=True)
class RenderedAttack:
    attack: str
    text: str
    image_path: str
    bbox: tuple[int, int, int, int] | None
    target_wrong_label: str | None
    violation_type: str | None
    placement: str

    def to_dict(self) -> dict:
        return asdict(self)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def render_attack(source: str, spec: AttackText, output: Path) -> RenderedAttack:
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(source).convert("RGB")
    if max(image.size) < 512:
        scale = 512 / max(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    if spec.kind == "none":
        image.save(output, quality=95)
        return RenderedAttack(spec.kind, "", str(output.resolve()), None, None, None, "none")

    draw = ImageDraw.Draw(image)
    font_size = max(18, min(34, image.width // 22))
    font = _font(font_size)
    max_chars = max(18, int(image.width / (font_size * 0.58)))
    lines = textwrap.wrap(spec.text, width=max_chars) or [spec.text]
    line_gap = 5
    heights = [draw.textbbox((0, 0), line, font=font, stroke_width=1)[3] for line in lines]
    box_h = sum(heights) + line_gap * (len(lines) - 1) + 20

    if spec.kind == "naive":
        x0, y0, x1, y1 = 0, 0, image.width, min(image.height, box_h)
        fill, text_fill, placement = (255, 255, 255), (220, 0, 0), "top-banner"
    else:
        margin = max(8, image.width // 80)
        x0, y0, x1, y1 = margin, max(margin, image.height - box_h - margin), image.width - margin, image.height - margin
        fill, text_fill, placement = (245, 240, 215), (28, 28, 28), "scene-caption-plaque"
    draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=fill, outline=(25, 25, 25), width=2)
    y = y0 + 10
    for line, height in zip(lines, heights):
        draw.text((x0 + 10, y), line, font=font, fill=text_fill, stroke_width=1, stroke_fill=fill)
        y += height + line_gap
    image.save(output, quality=95)
    return RenderedAttack(
        spec.kind, spec.text, str(output.resolve()), (x0, y0, x1, y1), spec.target_wrong_label,
        spec.violation_type, placement,
    )


def mask_bbox(image_path: str, bbox: tuple[int, int, int, int] | None, output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    if bbox is not None:
        draw = ImageDraw.Draw(image)
        draw.rectangle(bbox, fill=(127, 127, 127))
    image.save(output, quality=95)
    return str(output.resolve())

