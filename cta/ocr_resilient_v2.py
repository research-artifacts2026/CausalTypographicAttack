from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageStat

from cta.ocr_resilient import (
    OCRResilientRender,
    _fit_text,
    _font,
    _intersection_fraction,
    _line_positions,
    _sha256,
)


@dataclass(frozen=True)
class OCRResilientStyleV2:
    style_id: str
    display_name: str
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


STYLES_V2 = (
    OCRResilientStyleV2("dot-matrix", "Dot matrix", "complete claim drawn as separated LED-like dots"),
    OCRResilientStyleV2("dot-matrix-chroma", "Chromatic dot matrix", "dot lettering using approximately isoluminant opposing colors"),
    OCRResilientStyleV2("isoluminant-caption", "Isoluminant caption", "solid glyph shapes carried mainly by chromatic rather than luminance contrast"),
    OCRResilientStyleV2("checkerboard-glyph", "Checkerboard glyph", "human-readable glyphs with half of their stroke pixels deterministically removed"),
    OCRResilientStyleV2("jittered-characters", "Jittered characters", "character-wise baseline perturbation while preserving word order"),
    OCRResilientStyleV2("arced-characters", "Arced characters", "character-wise sinusoidal placement resembling curved scene lettering"),
    OCRResilientStyleV2("negative-dot-matrix", "Negative dot matrix", "the claim appears as source-texture holes in a translucent local tint"),
    OCRResilientStyleV2("microtype-repeat", "Microtype repeat", "two complete small copies with different chromatic channels"),
)
STYLE_V2_BY_ID = {style.style_id: style for style in STYLES_V2}


def style_ids_v2() -> list[str]:
    return [style.style_id for style in STYLES_V2]


def _standard_mask(size: tuple[int, int], positions, font: ImageFont.ImageFont) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for line, x, y in positions:
        draw.text((x, y), line, font=font, fill=255)
    return mask


def _mask_to_dots(mask: Image.Image, block: int = 3) -> Image.Image:
    source = np.asarray(mask)
    height, width = source.shape
    result = Image.new("L", mask.size, 0)
    draw = ImageDraw.Draw(result)
    radius = max(1, block // 2)
    for y in range(0, height, block):
        for x in range(0, width, block):
            tile = source[y:min(height, y + block), x:min(width, x + block)]
            if tile.size and int(tile.max()) >= 96:
                cx = min(width - 1, x + block // 2)
                cy = min(height - 1, y + block // 2)
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
    return result


def _composite_color(patch: Image.Image, glyph_mask: Image.Image, color: tuple[int, int, int, int]) -> Image.Image:
    layer = Image.new("RGBA", patch.size, color)
    layer.putalpha(Image.fromarray(
        (np.asarray(glyph_mask, dtype=np.float32) * (color[3] / 255.0)).astype(np.uint8),
        mode="L",
    ))
    return Image.alpha_composite(patch.convert("RGBA"), layer)


def _isoluminant_pair(patch: Image.Image) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    luminance = float(ImageStat.Stat(patch.convert("L")).mean[0])
    if luminance >= 120:
        # Perceptually distinct purple/green pair with similar luma.
        return (156, 76, 174), (44, 132, 76)
    return (235, 92, 70), (38, 155, 178)


def _draw_characterwise(
    patch: Image.Image,
    text: str,
    *,
    arc: bool,
) -> tuple[Image.Image, Image.Image]:
    font, lines, line_height = _fit_text(text, patch.width, patch.height)
    foreground = (246, 249, 252, 225)
    shadow = (8, 12, 18, 175)
    layer = Image.new("RGBA", patch.size, (0, 0, 0, 0))
    logical = Image.new("L", patch.size, 0)
    draw = ImageDraw.Draw(layer)
    mask_draw = ImageDraw.Draw(logical)
    total_height = len(lines) * line_height
    base_y = max(2, (patch.height - total_height) // 2)
    for line_index, line in enumerate(lines):
        widths = [font.getlength(char) for char in line]
        x = max(2.0, (patch.width - sum(widths)) / 2.0)
        for char_index, (char, char_width) in enumerate(zip(line, widths)):
            if arc:
                phase = (char_index / max(1, len(line) - 1)) * math.pi
                offset = round(5.0 * math.sin(phase)) - 2
            else:
                offset = (-3, 2, 0, 4, -1)[(char_index + line_index) % 5]
            y = base_y + line_index * line_height + offset
            draw.text((x + 1, y + 1), char, font=font, fill=shadow)
            draw.text((x, y), char, font=font, fill=foreground)
            mask_draw.text((x, y), char, font=font, fill=255)
            x += char_width
    return Image.alpha_composite(patch.convert("RGBA"), layer), logical


def _render_patch_v2(patch: Image.Image, text: str, style_id: str) -> tuple[Image.Image, Image.Image]:
    patch = patch.convert("RGBA")
    font, lines, line_height = _fit_text(text, patch.width, patch.height)
    positions = _line_positions(lines, font, line_height, patch.width, patch.height)
    standard = _standard_mask(patch.size, positions, font)

    if style_id == "dot-matrix":
        dots = _mask_to_dots(standard, block=3)
        return _composite_color(patch, dots, (250, 245, 125, 232)), dots
    if style_id == "dot-matrix-chroma":
        dots = _mask_to_dots(standard, block=3)
        foreground, tint = _isoluminant_pair(patch)
        tinted = Image.blend(patch, Image.new("RGBA", patch.size, (*tint, 255)), 0.20)
        return _composite_color(tinted, dots, (*foreground, 215)), dots
    if style_id == "isoluminant-caption":
        foreground, tint = _isoluminant_pair(patch)
        tinted = Image.blend(patch, Image.new("RGBA", patch.size, (*tint, 255)), 0.26)
        return _composite_color(tinted, standard, (*foreground, 210)), standard
    if style_id == "checkerboard-glyph":
        source = np.asarray(standard).copy()
        yy, xx = np.mgrid[:source.shape[0], :source.shape[1]]
        source[((xx + yy) % 4) >= 2] = 0
        checker = Image.fromarray(source.astype(np.uint8), mode="L")
        return _composite_color(patch, checker, (250, 250, 250, 240)), checker
    if style_id == "jittered-characters":
        return _draw_characterwise(patch, text, arc=False)
    if style_id == "arced-characters":
        return _draw_characterwise(patch, text, arc=True)
    if style_id == "negative-dot-matrix":
        dots = _mask_to_dots(standard, block=3)
        tint = Image.new("RGBA", patch.size, (15, 35, 52, 150))
        tint_alpha = np.full((patch.height, patch.width), 150, dtype=np.uint8)
        tint_alpha[np.asarray(dots) > 0] = 0
        tint.putalpha(Image.fromarray(tint_alpha, mode="L"))
        return Image.alpha_composite(patch, tint), dots
    if style_id == "microtype-repeat":
        micro_size = max(8, int(getattr(font, "size", 12) * 0.72))
        micro_font = _font(micro_size, bold=True)
        micro_lines = []
        words = text.split()
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if micro_font.getlength(candidate) <= patch.width - 8:
                current = candidate
            else:
                if current:
                    micro_lines.append(current)
                current = word
        if current:
            micro_lines.append(current)
        micro_height = micro_font.getbbox("Ag")[3] - micro_font.getbbox("Ag")[1] + 1
        one_copy_height = len(micro_lines) * micro_height
        if one_copy_height * 2 + 3 > patch.height:
            # The complete claim remains present once when a very shallow
            # SceneTAP region cannot hold two copies.
            copies = 1
        else:
            copies = 2
        layer = Image.new("RGBA", patch.size, (0, 0, 0, 0))
        logical = Image.new("L", patch.size, 0)
        draw = ImageDraw.Draw(layer)
        mask_draw = ImageDraw.Draw(logical)
        y = max(1, (patch.height - one_copy_height * copies - max(0, copies - 1) * 3) // 2)
        colors = ((255, 118, 92, 225), (76, 205, 236, 215))
        for copy_index in range(copies):
            for line in micro_lines:
                x = max(3.0, (patch.width - micro_font.getlength(line)) / 2.0)
                draw.text((x, y), line, font=micro_font, fill=colors[copy_index])
                mask_draw.text((x, y), line, font=micro_font, fill=255)
                y += micro_height
            y += 3
        return Image.alpha_composite(patch, layer), logical
    raise ValueError(f"unknown v2 style: {style_id}")


def render_ocr_resilient_carrier_v2(
    *,
    scenetap_image: str,
    clean_image: str,
    attack_text: str,
    layout_bbox: tuple[int, int, int, int] | list[int],
    target_bbox: tuple[float, float, float, float] | list[float],
    style_id: str,
    output: Path,
    carrier_mask_output: Path,
    max_image_overlay_fraction: float = 0.18,
    max_object_occlusion_fraction: float = 0.32,
) -> OCRResilientRender:
    if style_id not in STYLE_V2_BY_ID:
        raise ValueError(f"unknown v2 style_id: {style_id}")
    if output.suffix.lower() != ".png" or carrier_mask_output.suffix.lower() != ".png":
        raise ValueError("lossless PNG output is required")
    base = Image.open(scenetap_image).convert("RGB")
    clean = Image.open(clean_image).convert("RGB")
    if clean.size != base.size:
        clean = clean.resize(base.size, Image.Resampling.LANCZOS)
    bbox = tuple(int(value) for value in layout_bbox)
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= base.width and 0 <= y0 < y1 <= base.height):
        raise ValueError("layout_bbox falls outside image")
    target = tuple(float(value) for value in target_bbox)
    area = (x1 - x0) * (y1 - y0) / (base.width * base.height)
    occlusion = _intersection_fraction(bbox, target)
    if area > max_image_overlay_fraction + 1e-12:
        raise ValueError("SceneTAP layout area exceeds cap")
    if occlusion > max_object_occlusion_fraction + 1e-12:
        raise ValueError("SceneTAP layout occlusion exceeds cap")
    carrier_patch, carrier_mask = _render_patch_v2(clean.crop(bbox), attack_text, style_id)
    rendered = base.copy()
    rendered.paste(carrier_patch.convert("RGB"), (x0, y0))
    full_mask = Image.new("L", base.size, 0)
    full_mask.paste(carrier_mask, (x0, y0))
    output.parent.mkdir(parents=True, exist_ok=True)
    carrier_mask_output.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output, format="PNG", optimize=False)
    full_mask.save(carrier_mask_output, format="PNG", optimize=False)
    before = np.asarray(base)
    after = np.asarray(rendered)
    changed = np.any(before != after, axis=2)
    inside = np.zeros(changed.shape, dtype=bool)
    inside[y0:y1, x0:x1] = True
    outside = int(np.count_nonzero(changed & ~inside))
    if outside:
        raise AssertionError("pixels changed outside SceneTAP bbox")
    carrier_pixels = int(np.count_nonzero(np.asarray(full_mask) > 0))
    if not carrier_pixels:
        raise AssertionError("carrier mask is empty")
    return OCRResilientRender(
        image_path=str(output.resolve()),
        carrier_mask_path=str(carrier_mask_output.resolve()),
        rendered_sha256=_sha256(output),
        carrier_mask_sha256=_sha256(carrier_mask_output),
        source_dimensions=base.size,
        rendered_dimensions=rendered.size,
        layout_bbox=bbox,
        layout_area_fraction=area,
        target_bbox=target,
        object_bbox_occlusion_fraction=occlusion,
        changed_pixels_outside_layout_bbox=outside,
        outside_layout_bbox_unchanged=outside == 0,
        carrier_pixel_count=carrier_pixels,
        attack_text=attack_text,
        style=STYLE_V2_BY_ID[style_id].to_dict(),
        integration="SceneTAP-planned region with defense-aware nonstandard glyph carrier v2",
    )
