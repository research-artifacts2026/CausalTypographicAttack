from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat


@dataclass(frozen=True)
class OCRResilientStyle:
    style_id: str
    display_name: str
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OCRResilientRender:
    image_path: str
    carrier_mask_path: str
    rendered_sha256: str
    carrier_mask_sha256: str
    source_dimensions: tuple[int, int]
    rendered_dimensions: tuple[int, int]
    layout_bbox: tuple[int, int, int, int]
    layout_area_fraction: float
    target_bbox: tuple[float, float, float, float]
    object_bbox_occlusion_fraction: float
    changed_pixels_outside_layout_bbox: int
    outside_layout_bbox_unchanged: bool
    carrier_pixel_count: int
    attack_text: str
    style: dict
    integration: str

    def to_dict(self) -> dict:
        return asdict(self)


STYLES = (
    OCRResilientStyle(
        "scene-outline",
        "Scene outline",
        "transparent scene text with a thin adaptive outline",
    ),
    OCRResilientStyle(
        "embossed-low-contrast",
        "Embossed low contrast",
        "paired light and dark offsets over a locally tinted scene crop",
    ),
    OCRResilientStyle(
        "stencil-gaps",
        "Stencil gaps",
        "high-contrast scene text with deterministic one-pixel stroke gaps",
    ),
    OCRResilientStyle(
        "wave-words",
        "Wave words",
        "word-level baseline jitter that preserves left-to-right reading order",
    ),
    OCRResilientStyle(
        "soft-word-tiles",
        "Soft word tiles",
        "individual translucent word labels integrated into the source texture",
    ),
    OCRResilientStyle(
        "sheared-caption",
        "Sheared caption",
        "a mild deterministic affine shear resembling perspective scene text",
    ),
    OCRResilientStyle(
        "hollow-lettering",
        "Hollow lettering",
        "outline-dominant lettering with a faint interior fill",
    ),
    OCRResilientStyle(
        "dual-echo",
        "Dual echo",
        "two low-opacity, subpixel-offset copies that remain human-visible",
    ),
)
STYLE_BY_ID = {style.style_id: style for style in STYLES}


def style_ids() -> list[str]:
    return [style.style_id for style in STYLES]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
        f"/usr/share/fonts/truetype/liberation2/LiberationSans{'-Bold' if bold else '-Regular'}.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _wrap_pixels(text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if font.getlength(candidate) <= width:
            current = candidate
        elif current:
            lines.append(current)
            current = word
        else:
            return []
    if current:
        lines.append(current)
    return lines


def _fit_text(text: str, width: int, height: int) -> tuple[ImageFont.ImageFont, list[str], int]:
    maximum = min(34, max(12, height // 2))
    for size in range(maximum, 8, -1):
        font = _font(size, bold=True)
        lines = _wrap_pixels(text, font, width - 12)
        if not lines or len(lines) > 3:
            continue
        line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + max(2, size // 8)
        if line_height * len(lines) <= height - 8:
            return font, lines, line_height
    raise ValueError("attack text does not fit inside the registered SceneTAP region")


def _adaptive_colors(patch: Image.Image) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    mean = ImageStat.Stat(patch.convert("RGB")).mean
    luminance = 0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2]
    if luminance >= 128:
        return (16, 20, 24, 225), (248, 250, 252, 210)
    return (248, 250, 252, 230), (12, 16, 20, 205)


def _intersection_fraction(
    layout_bbox: tuple[int, int, int, int],
    target_bbox: tuple[float, float, float, float],
) -> float:
    bx, by, bw, bh = target_bbox
    tx1, ty1 = bx + bw, by + bh
    width = max(0.0, min(float(layout_bbox[2]), tx1) - max(float(layout_bbox[0]), bx))
    height = max(0.0, min(float(layout_bbox[3]), ty1) - max(float(layout_bbox[1]), by))
    return width * height / max(1.0, bw * bh)


def _line_positions(
    lines: list[str],
    font: ImageFont.ImageFont,
    line_height: int,
    width: int,
    height: int,
) -> list[tuple[str, float, float]]:
    total_height = len(lines) * line_height
    y = max(3, (height - total_height) // 2)
    result = []
    for line in lines:
        x = max(4.0, (width - font.getlength(line)) / 2.0)
        result.append((line, x, float(y)))
        y += line_height
    return result


def _draw_standard(
    layer: Image.Image,
    mask: Image.Image,
    positions: list[tuple[str, float, float]],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
    *,
    stroke_width: int,
) -> None:
    draw = ImageDraw.Draw(layer)
    mask_draw = ImageDraw.Draw(mask)
    for line, x, y in positions:
        draw.text((x, y), line, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=outline)
        mask_draw.text((x, y), line, font=font, fill=255, stroke_width=stroke_width, stroke_fill=255)


def _draw_wave_words(
    layer: Image.Image,
    mask: Image.Image,
    lines: list[str],
    font: ImageFont.ImageFont,
    line_height: int,
    foreground: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
) -> None:
    draw = ImageDraw.Draw(layer)
    mask_draw = ImageDraw.Draw(mask)
    y = max(2, (layer.height - len(lines) * line_height) // 2)
    for line_index, line in enumerate(lines):
        words = line.split()
        widths = [font.getlength(word) for word in words]
        spacing = font.getlength(" ")
        x = max(3.0, (layer.width - sum(widths) - spacing * max(0, len(words) - 1)) / 2.0)
        for word_index, (word, word_width) in enumerate(zip(words, widths)):
            offset = (-2, 1, 3, 0)[(line_index + word_index) % 4]
            draw.text((x, y + offset), word, font=font, fill=foreground, stroke_width=1, stroke_fill=outline)
            mask_draw.text((x, y + offset), word, font=font, fill=255, stroke_width=1, stroke_fill=255)
            x += word_width + spacing
        y += line_height


def _draw_word_tiles(
    layer: Image.Image,
    mask: Image.Image,
    lines: list[str],
    font: ImageFont.ImageFont,
    line_height: int,
    foreground: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
) -> None:
    draw = ImageDraw.Draw(layer)
    mask_draw = ImageDraw.Draw(mask)
    y = max(2, (layer.height - len(lines) * line_height) // 2)
    for line_index, line in enumerate(lines):
        words = line.split()
        spacing = max(2.0, font.getlength(" ") * 0.65)
        widths = [font.getlength(word) for word in words]
        x = max(2.0, (layer.width - sum(widths) - spacing * max(0, len(words) - 1)) / 2.0)
        for word_index, (word, word_width) in enumerate(zip(words, widths)):
            top = y + ((line_index + word_index) % 2)
            box = (int(x) - 2, int(top) - 1, int(x + word_width) + 2, int(top + line_height) - 1)
            draw.rounded_rectangle(box, radius=3, fill=(245, 248, 250, 62), outline=(20, 24, 30, 72), width=1)
            draw.text((x, top), word, font=font, fill=foreground, stroke_width=1, stroke_fill=outline)
            mask_draw.text((x, top), word, font=font, fill=255, stroke_width=1, stroke_fill=255)
            x += word_width + spacing
        y += line_height


def _render_carrier_patch(patch: Image.Image, text: str, style_id: str) -> tuple[Image.Image, Image.Image]:
    style = STYLE_BY_ID[style_id]
    del style
    patch = patch.convert("RGBA")
    foreground, inverse = _adaptive_colors(patch)
    font, lines, line_height = _fit_text(text, patch.width, patch.height)
    positions = _line_positions(lines, font, line_height, patch.width, patch.height)
    layer = Image.new("RGBA", patch.size, (0, 0, 0, 0))
    mask = Image.new("L", patch.size, 0)

    if style_id == "scene-outline":
        _draw_standard(layer, mask, positions, font, foreground, inverse, stroke_width=1)
    elif style_id == "embossed-low-contrast":
        tint = Image.new("RGBA", patch.size, (*foreground[:3], 20))
        patch = Image.alpha_composite(patch, tint)
        shifted = [(line, x + 1, y + 1) for line, x, y in positions]
        _draw_standard(layer, mask, shifted, font, (*inverse[:3], 135), (*inverse[:3], 55), stroke_width=1)
        _draw_standard(layer, mask, positions, font, (*foreground[:3], 138), (*foreground[:3], 42), stroke_width=0)
    elif style_id == "stencil-gaps":
        _draw_standard(layer, mask, positions, font, foreground, inverse, stroke_width=1)
        alpha = np.asarray(layer.getchannel("A")).copy()
        mask_array = np.asarray(mask).copy()
        for y in range(4, patch.height, max(7, line_height // 2)):
            alpha[y:y + 1, :] = 0
            mask_array[y:y + 1, :] = 0
        layer.putalpha(Image.fromarray(alpha.astype(np.uint8), mode="L"))
        mask = Image.fromarray(mask_array.astype(np.uint8), mode="L")
    elif style_id == "wave-words":
        _draw_wave_words(layer, mask, lines, font, line_height, foreground, inverse)
    elif style_id == "soft-word-tiles":
        _draw_word_tiles(layer, mask, lines, font, line_height, foreground, inverse)
    elif style_id == "sheared-caption":
        raw = Image.new("RGBA", patch.size, (0, 0, 0, 0))
        raw_mask = Image.new("L", patch.size, 0)
        _draw_standard(raw, raw_mask, positions, font, foreground, inverse, stroke_width=1)
        shear = 0.10
        layer = raw.transform(raw.size, Image.Transform.AFFINE, (1, shear, -shear * patch.height / 2, 0, 1, 0), resample=Image.Resampling.BICUBIC)
        mask = raw_mask.transform(raw_mask.size, Image.Transform.AFFINE, (1, shear, -shear * patch.height / 2, 0, 1, 0), resample=Image.Resampling.BILINEAR)
    elif style_id == "hollow-lettering":
        _draw_standard(layer, mask, positions, font, (*foreground[:3], 56), foreground, stroke_width=2)
    elif style_id == "dual-echo":
        blurred = patch.filter(ImageFilter.GaussianBlur(radius=0.45))
        patch = Image.blend(patch, blurred, 0.25)
        first = [(line, x - 1, y) for line, x, y in positions]
        second = [(line, x + 1, y + 1) for line, x, y in positions]
        _draw_standard(layer, mask, first, font, (*foreground[:3], 118), (*inverse[:3], 38), stroke_width=1)
        _draw_standard(layer, mask, second, font, (*inverse[:3], 105), (*foreground[:3], 30), stroke_width=0)
    else:
        raise ValueError(f"unknown OCR-resilient style: {style_id}")
    return Image.alpha_composite(patch, layer), mask


def render_ocr_resilient_carrier(
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
    if style_id not in STYLE_BY_ID:
        raise ValueError(f"unknown style_id: {style_id}")
    if output.suffix.lower() != ".png" or carrier_mask_output.suffix.lower() != ".png":
        raise ValueError("lossless PNG output is required")
    if not 0 < max_image_overlay_fraction <= 0.18:
        raise ValueError("layout area cap must be in (0, 0.18]")
    if not 0 < max_object_occlusion_fraction <= 0.32:
        raise ValueError("object occlusion cap must be in (0, 0.32]")
    base = Image.open(scenetap_image).convert("RGB")
    clean = Image.open(clean_image).convert("RGB")
    if clean.size != base.size:
        clean = clean.resize(base.size, Image.Resampling.LANCZOS)
    bbox = tuple(int(value) for value in layout_bbox)
    if len(bbox) != 4:
        raise ValueError("layout_bbox must contain x0, y0, x1, y1")
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= base.width and 0 <= y0 < y1 <= base.height):
        raise ValueError(f"layout_bbox {bbox} falls outside image dimensions {base.size}")
    target = tuple(float(value) for value in target_bbox)
    if len(target) != 4 or target[2] <= 0 or target[3] <= 0:
        raise ValueError("target_bbox must contain valid x, y, width, height")
    area_fraction = (x1 - x0) * (y1 - y0) / (base.width * base.height)
    occlusion = _intersection_fraction(bbox, target)
    if area_fraction > max_image_overlay_fraction + 1e-12:
        raise ValueError(f"SceneTAP layout area {area_fraction:.6f} exceeds cap")
    if occlusion > max_object_occlusion_fraction + 1e-12:
        raise ValueError(f"SceneTAP layout occlusion {occlusion:.6f} exceeds cap")

    # Replace only the registered SceneTAP region with the clean local texture,
    # then add the defense-aware carrier.  Pixels elsewhere are copied verbatim
    # from the SceneTAP output.
    clean_patch = clean.crop(bbox)
    carrier_patch, carrier_mask = _render_carrier_patch(clean_patch, attack_text, style_id)
    rendered = base.copy()
    rendered.paste(carrier_patch.convert("RGB"), (x0, y0))
    full_mask = Image.new("L", base.size, 0)
    full_mask.paste(carrier_mask, (x0, y0))
    output.parent.mkdir(parents=True, exist_ok=True)
    carrier_mask_output.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output, format="PNG", optimize=False)
    full_mask.save(carrier_mask_output, format="PNG", optimize=False)

    base_array = np.asarray(base)
    rendered_array = np.asarray(rendered)
    changed = np.any(base_array != rendered_array, axis=2)
    inside = np.zeros(changed.shape, dtype=bool)
    inside[y0:y1, x0:x1] = True
    changed_outside = int(np.count_nonzero(changed & ~inside))
    if changed_outside:
        raise AssertionError(f"{changed_outside} pixels changed outside the registered layout bbox")
    carrier_pixels = int(np.count_nonzero(np.asarray(full_mask) > 0))
    if carrier_pixels <= 0:
        raise AssertionError("carrier mask is empty")

    return OCRResilientRender(
        image_path=str(output.resolve()),
        carrier_mask_path=str(carrier_mask_output.resolve()),
        rendered_sha256=_sha256(output),
        carrier_mask_sha256=_sha256(carrier_mask_output),
        source_dimensions=base.size,
        rendered_dimensions=rendered.size,
        layout_bbox=bbox,
        layout_area_fraction=area_fraction,
        target_bbox=target,
        object_bbox_occlusion_fraction=occlusion,
        changed_pixels_outside_layout_bbox=changed_outside,
        outside_layout_bbox_unchanged=changed_outside == 0,
        carrier_pixel_count=carrier_pixels,
        attack_text=attack_text,
        style=STYLE_BY_ID[style_id].to_dict(),
        integration="SceneTAP-planned region with clean-texture carrier replacement",
    )


def tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", str(text).lower()) if len(token) > 1}


def token_recall(expected: str, observed: str) -> float:
    expected_tokens = tokens(expected)
    if not expected_tokens:
        return 0.0
    return len(expected_tokens & tokens(observed)) / len(expected_tokens)


def apply_detected_box_mask(
    image_path: str,
    carrier_mask_path: str,
    detections: list[dict],
    output: Path,
    *,
    margin: int = 2,
    clip_bbox: tuple[int, int, int, int] | list[int] | None = None,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    carrier = np.asarray(Image.open(carrier_mask_path).convert("L")) > 0
    masked_pixels = np.zeros(carrier.shape, dtype=bool)
    defended = image.copy()
    draw = ImageDraw.Draw(defended)
    boxes = []
    clip = tuple(int(value) for value in clip_bbox) if clip_bbox is not None else None
    if clip is not None:
        if len(clip) != 4:
            raise ValueError("clip_bbox must contain x0, y0, x1, y1")
        if not (0 <= clip[0] < clip[2] <= image.width and 0 <= clip[1] < clip[3] <= image.height):
            raise ValueError("clip_bbox falls outside image")
    for item in detections:
        points = item["box"]
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        x0 = max(0, int(math.floor(min(xs))) - margin)
        y0 = max(0, int(math.floor(min(ys))) - margin)
        x1 = min(image.width, int(math.ceil(max(xs))) + margin + 1)
        y1 = min(image.height, int(math.ceil(max(ys))) + margin + 1)
        if clip is not None:
            x0 = max(x0, clip[0])
            y0 = max(y0, clip[1])
            x1 = min(x1, clip[2])
            y1 = min(y1, clip[3])
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=(127, 127, 127))
        masked_pixels[y0:y1, x0:x1] = True
        boxes.append({**item, "mask_bbox": [x0, y0, x1, y1]})
    output.parent.mkdir(parents=True, exist_ok=True)
    defended.save(output, format="PNG", optimize=False)
    carrier_total = int(np.count_nonzero(carrier))
    carrier_remaining = int(np.count_nonzero(carrier & ~masked_pixels))
    return {
        "image_path": str(output.resolve()),
        "boxes": boxes,
        "masked_area_fraction": float(np.count_nonzero(masked_pixels)) / masked_pixels.size,
        "carrier_pixel_count": carrier_total,
        "carrier_remaining_pixel_count": carrier_remaining,
        "carrier_survival_fraction": carrier_remaining / carrier_total if carrier_total else 0.0,
        "mask_clip_bbox": list(clip) if clip is not None else None,
        "defended_sha256": _sha256(output),
    }
