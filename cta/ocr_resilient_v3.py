from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

from cta.ocr_resilient import (
    OCRResilientRender,
    _fit_text,
    _font,
    _intersection_fraction,
    _line_positions,
    _sha256,
)


@dataclass(frozen=True)
class OCRResilientStyleV3:
    style_id: str
    display_name: str
    description: str
    alpha: int

    def to_dict(self) -> dict:
        return asdict(self)


STYLES_V3 = (
    OCRResilientStyleV3("chroma-a40", "Chroma alpha 40", "standard glyphs at very low chromatic opacity", 40),
    OCRResilientStyleV3("chroma-a56", "Chroma alpha 56", "standard glyphs at low chromatic opacity", 56),
    OCRResilientStyleV3("chroma-a72", "Chroma alpha 72", "standard glyphs at medium-low chromatic opacity", 72),
    OCRResilientStyleV3("chroma-a88", "Chroma alpha 88", "standard glyphs at medium chromatic opacity", 88),
    OCRResilientStyleV3("chroma-blur72", "Soft chroma 72", "standard glyphs with a 0.65 px soft edge", 72),
    OCRResilientStyleV3("chroma-stripe88", "Striped chroma 88", "standard glyphs with alternating horizontal stroke bands", 88),
    OCRResilientStyleV3("thin-chroma88", "Thin chroma 88", "regular-weight standard glyphs with chromatic contrast", 88),
    OCRResilientStyleV3("dual-chroma56", "Dual chroma 56", "two faint opposing-color copies offset by one pixel", 56),
)
STYLE_V3_BY_ID = {style.style_id: style for style in STYLES_V3}


def style_ids_v3() -> list[str]:
    return [style.style_id for style in STYLES_V3]


def _pair(patch: Image.Image) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    luminance = float(ImageStat.Stat(patch.convert("L")).mean[0])
    if luminance >= 120:
        return (230, 38, 54), (10, 88, 34)
    return (255, 76, 52), (12, 112, 128)


def _standard_mask(
    patch: Image.Image,
    text: str,
    *,
    regular: bool = False,
) -> tuple[Image.Image, ImageFont.ImageFont]:
    font, lines, line_height = _fit_text(text, patch.width, patch.height)
    if regular:
        font = _font(getattr(font, "size", 12), bold=False)
        # Preserve the already fitted line breaks; regular weight is narrower.
    positions = _line_positions(lines, font, line_height, patch.width, patch.height)
    mask = Image.new("L", patch.size, 0)
    draw = ImageDraw.Draw(mask)
    for line, x, y in positions:
        draw.text((x, y), line, font=font, fill=255)
    return mask, font


def _paint(patch: Image.Image, mask: Image.Image, color: tuple[int, int, int], alpha: int) -> Image.Image:
    layer = Image.new("RGBA", patch.size, (*color, alpha))
    layer.putalpha(Image.fromarray(
        (np.asarray(mask, dtype=np.float32) * (alpha / 255.0)).astype(np.uint8),
        mode="L",
    ))
    return Image.alpha_composite(patch.convert("RGBA"), layer)


def _render_patch_v3(patch: Image.Image, text: str, style_id: str) -> tuple[Image.Image, Image.Image]:
    style = STYLE_V3_BY_ID[style_id]
    foreground, background_tint = _pair(patch)
    # A weak scene-colored tint suppresses luminance edges without replacing
    # the source texture with a solid card.
    tinted = Image.blend(
        patch.convert("RGBA"),
        Image.new("RGBA", patch.size, (*background_tint, 255)),
        0.08,
    )
    regular = style_id == "thin-chroma88"
    mask, _ = _standard_mask(tinted, text, regular=regular)
    if style_id == "chroma-blur72":
        mask = mask.filter(ImageFilter.GaussianBlur(radius=0.65))
    elif style_id == "chroma-stripe88":
        array = np.asarray(mask).copy()
        for y in range(1, array.shape[0], 4):
            array[y:y + 2, :] = 0
        mask = Image.fromarray(array.astype(np.uint8), mode="L")
    if style_id == "dual-chroma56":
        first = _paint(tinted, mask, foreground, style.alpha)
        shifted = Image.new("L", mask.size, 0)
        shifted.paste(mask, (1, 1))
        logical = Image.fromarray(np.maximum(np.asarray(mask), np.asarray(shifted)).astype(np.uint8), mode="L")
        return _paint(first, shifted, background_tint, style.alpha), logical
    return _paint(tinted, mask, foreground, style.alpha), mask


def render_ocr_resilient_carrier_v3(
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
    if style_id not in STYLE_V3_BY_ID:
        raise ValueError(f"unknown v3 style_id: {style_id}")
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
    carrier_patch, carrier_mask = _render_patch_v3(clean.crop(bbox), attack_text, style_id)
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
        style=STYLE_V3_BY_ID[style_id].to_dict(),
        integration="SceneTAP-planned region with low-luminance-contrast standard glyph carrier v3",
    )
