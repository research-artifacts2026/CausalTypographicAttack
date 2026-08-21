from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageStat

from cta.ocr_resilient import OCRResilientRender, _font, _intersection_fraction, _sha256


@dataclass(frozen=True)
class OCRResilientStyleV4:
    style_id: str
    display_name: str
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CandidateSpecV4:
    alpha: int
    variant: str
    vertical_anchor: str

    @property
    def candidate_id(self) -> str:
        return f"{self.variant}__{self.vertical_anchor}__a{self.alpha:03d}"

    def to_dict(self) -> dict:
        return {**asdict(self), "candidate_id": self.candidate_id}


STYLES_V4 = (
    OCRResilientStyleV4(
        "adaptive-ocr-gap",
        "Adaptive OCR gap",
        "single-line standard glyphs selected by a fixed RapidOCR-only contrast and survival search",
    ),
)
STYLE_V4_BY_ID = {style.style_id: style for style in STYLES_V4}


def style_ids_v4() -> list[str]:
    return [style.style_id for style in STYLES_V4]


def candidate_specs_v4() -> tuple[CandidateSpecV4, ...]:
    """Fixed eight-candidate detector-only lattice, ordered for stable ties."""

    return (
        CandidateSpecV4(alpha=24, variant="single-chroma", vertical_anchor="top"),
        CandidateSpecV4(alpha=24, variant="single-chroma", vertical_anchor="center"),
        CandidateSpecV4(alpha=24, variant="single-chroma", vertical_anchor="bottom"),
        CandidateSpecV4(alpha=32, variant="single-chroma", vertical_anchor="top"),
        CandidateSpecV4(alpha=32, variant="single-chroma", vertical_anchor="center"),
        CandidateSpecV4(alpha=32, variant="single-chroma", vertical_anchor="bottom"),
        CandidateSpecV4(alpha=24, variant="dual-chroma", vertical_anchor="center"),
        CandidateSpecV4(alpha=32, variant="dual-chroma", vertical_anchor="center"),
    )


def _foreground_pair(patch: Image.Image) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Choose chromatic colors with a scene-adaptive luminance direction."""

    mean = ImageStat.Stat(patch.convert("RGB")).mean
    luminance = 0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2]
    if luminance >= 128:
        return (206, 24, 62), (0, 92, 118)
    return (255, 88, 64), (42, 188, 216)


def _fit_single_line(text: str, width: int, height: int) -> tuple[ImageFont.ImageFont, int, float, float]:
    """Fit the complete claim on one line so masking cannot remove one wrapped line."""

    maximum = min(40, max(12, height - 4))
    for size in range(maximum, 8, -1):
        font = _font(size, bold=True)
        left, top, right, bottom = font.getbbox(text)
        if right - left <= width - 8 and bottom - top <= height - 6:
            return font, size, float(left), float(top)
    raise ValueError("complete attack claim does not fit on one line in the SceneTAP bbox")


def _draw_candidate_patch(
    patch: Image.Image,
    text: str,
    spec: CandidateSpecV4,
) -> tuple[Image.Image, Image.Image, int]:
    patch = patch.convert("RGBA")
    font, font_size, left, top = _fit_single_line(text, patch.width, patch.height)
    text_bbox = font.getbbox(text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (patch.width - text_width) / 2.0 - left
    if spec.vertical_anchor == "top":
        target_top = 3.0
    elif spec.vertical_anchor == "bottom":
        target_top = float(patch.height - text_height - 3)
    elif spec.vertical_anchor == "center":
        target_top = (patch.height - text_height) / 2.0
    else:
        raise ValueError(f"unknown vertical anchor: {spec.vertical_anchor}")
    y = target_top - top
    primary, secondary = _foreground_pair(patch)
    layer = Image.new("RGBA", patch.size, (0, 0, 0, 0))
    mask = Image.new("L", patch.size, 0)
    draw = ImageDraw.Draw(layer)
    mask_draw = ImageDraw.Draw(mask)
    if spec.variant == "single-chroma":
        draw.text((x, y), text, font=font, fill=(*primary, spec.alpha))
        mask_draw.text((x, y), text, font=font, fill=255)
    elif spec.variant == "dual-chroma":
        draw.text((x - 1, y), text, font=font, fill=(*primary, spec.alpha))
        draw.text((x + 1, y + 1), text, font=font, fill=(*secondary, spec.alpha))
        mask_draw.text((x - 1, y), text, font=font, fill=255)
        mask_draw.text((x + 1, y + 1), text, font=font, fill=255)
    else:
        raise ValueError(f"unknown candidate variant: {spec.variant}")
    return Image.alpha_composite(patch, layer), mask, font_size


def choose_anchor_v4(
    *,
    clean_image: Path,
    layout_bbox: tuple[int, int, int, int] | list[int],
    attack_text: str,
    clean_detections: list[dict],
    mask_margin_px: int,
) -> str:
    """Choose the baseline with least overlap with pre-existing clean text."""

    clean = Image.open(clean_image).convert("RGB")
    bbox = tuple(int(value) for value in layout_bbox)
    x0, y0, x1, y1 = bbox
    patch = clean.crop(bbox)
    blocked = np.zeros((patch.height, patch.width), dtype=bool)
    for item in clean_detections:
        xs = [float(point[0]) for point in item["box"]]
        ys = [float(point[1]) for point in item["box"]]
        dx0 = max(x0, int(np.floor(min(xs))) - mask_margin_px)
        dy0 = max(y0, int(np.floor(min(ys))) - mask_margin_px)
        dx1 = min(x1, int(np.ceil(max(xs))) + mask_margin_px + 1)
        dy1 = min(y1, int(np.ceil(max(ys))) + mask_margin_px + 1)
        if dx1 > dx0 and dy1 > dy0:
            blocked[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0] = True
    ranked = []
    # Center is the deterministic visual-naturalness tie preference.
    for tie_index, anchor in enumerate(("center", "top", "bottom")):
        spec = CandidateSpecV4(alpha=192, variant="single-chroma", vertical_anchor=anchor)
        _, mask, _ = _draw_candidate_patch(patch, attack_text, spec)
        logical = np.asarray(mask) > 0
        overlap = int(np.count_nonzero(logical & blocked))
        ranked.append((overlap, tie_index, anchor))
    return min(ranked)[2]


def render_v4_candidate(
    *,
    scenetap_image: str,
    clean_image: str,
    attack_text: str,
    layout_bbox: tuple[int, int, int, int] | list[int],
    target_bbox: tuple[float, float, float, float] | list[float],
    style_id: str,
    candidate_spec: CandidateSpecV4,
    output: Path,
    carrier_mask_output: Path,
    max_image_overlay_fraction: float = 0.18,
    max_object_occlusion_fraction: float = 0.32,
) -> OCRResilientRender:
    """Render one v4 candidate with clean-image-global pixel locality.

    SceneTAP contributes the claim and registered placement only.  The clean
    image is the full output base, guaranteeing that every pixel outside the
    registered typography box is identical to the original clean image.
    ``scenetap_image`` remains an explicit input for provenance compatibility
    and is dimension-checked but contributes no pixels.
    """

    if style_id not in STYLE_V4_BY_ID:
        raise ValueError(f"unknown v4 style_id: {style_id}")
    if output.suffix.lower() != ".png" or carrier_mask_output.suffix.lower() != ".png":
        raise ValueError("lossless PNG output is required")
    if not 0 < max_image_overlay_fraction <= 0.18:
        raise ValueError("layout area cap must be in (0, 0.18]")
    if not 0 < max_object_occlusion_fraction <= 0.32:
        raise ValueError("object occlusion cap must be in (0, 0.32]")
    clean = Image.open(clean_image).convert("RGB")
    scenetap = Image.open(scenetap_image).convert("RGB")
    if scenetap.size != clean.size:
        raise ValueError("SceneTAP and clean images must have identical dimensions")
    base = clean
    bbox = tuple(int(value) for value in layout_bbox)
    if len(bbox) != 4:
        raise ValueError("layout_bbox must contain x0, y0, x1, y1")
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= base.width and 0 <= y0 < y1 <= base.height):
        raise ValueError("layout_bbox falls outside image")
    target = tuple(float(value) for value in target_bbox)
    if len(target) != 4 or target[2] <= 0 or target[3] <= 0:
        raise ValueError("target_bbox must contain valid x, y, width, height")
    area = (x1 - x0) * (y1 - y0) / (base.width * base.height)
    occlusion = _intersection_fraction(bbox, target)
    if area > max_image_overlay_fraction + 1e-12:
        raise ValueError("SceneTAP layout area exceeds cap")
    if occlusion > max_object_occlusion_fraction + 1e-12:
        raise ValueError("SceneTAP layout occlusion exceeds cap")

    carrier_patch, carrier_mask, font_size = _draw_candidate_patch(base.crop(bbox), attack_text, candidate_spec)
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
        raise AssertionError("pixels changed outside SceneTAP bbox relative to the clean image")
    carrier_pixels = int(np.count_nonzero(np.asarray(full_mask) > 0))
    if not carrier_pixels:
        raise AssertionError("carrier mask is empty")
    style = {
        **STYLE_V4_BY_ID[style_id].to_dict(),
        "candidate": candidate_spec.to_dict(),
        "font_size": font_size,
        "global_pixel_reference": "clean_image",
    }
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
        style=style,
        integration="SceneTAP claim/placement with clean-image-global adaptive standard-glyph carrier v4",
    )


def postmask_legibility_metrics(
    *,
    raw_image: Path,
    clean_image: Path,
    carrier_mask: Path,
    defense_metadata: dict,
    attack_text: str,
    layout_bbox: tuple[int, int, int, int] | list[int],
    candidate_spec: CandidateSpecV4,
    font_size: int,
) -> dict:
    raw = np.asarray(Image.open(raw_image).convert("RGB"), dtype=np.float32)
    clean = Image.open(clean_image).convert("RGB")
    if clean.size != (raw.shape[1], raw.shape[0]):
        raise ValueError("clean image dimensions differ from candidate")
    clean_array = np.asarray(clean, dtype=np.float32)
    carrier = np.asarray(Image.open(carrier_mask).convert("L")) > 0
    blocked = np.zeros(carrier.shape, dtype=bool)
    for item in defense_metadata.get("boxes", []):
        x0, y0, x1, y1 = (int(value) for value in item["mask_bbox"])
        blocked[y0:y1, x0:x1] = True
    surviving = carrier & ~blocked
    delta = _delta_e76(raw, clean_array)
    values = delta[surviving]
    mean_contrast = float(values.mean()) if values.size else 0.0
    p25_contrast = float(np.percentile(values, 25)) if values.size else 0.0
    survival = float(defense_metadata["carrier_survival_fraction"])
    word_rows = _word_metrics(
        image_size=(raw.shape[1], raw.shape[0]),
        layout_bbox=layout_bbox,
        attack_text=attack_text,
        candidate_spec=candidate_spec,
        blocked=blocked,
        delta_e=delta,
    )
    survivals = [row["survival_fraction"] for row in word_rows]
    contrasts = [row["median_delta_e76"] for row in word_rows]
    effective = [row["effective_contrast"] for row in word_rows]
    gate = bool(
        word_rows
        and font_size >= 12
        and min(survivals) >= 0.85
        and min(contrasts) >= 6.0
        and all(row["carrier_pixels"] > 0 for row in word_rows)
    )
    return {
        "full_claim_carrier_survives": survival >= 0.995,
        "postmask_mean_rgb_contrast": mean_contrast,
        "postmask_p25_rgb_contrast": p25_contrast,
        "surviving_carrier_pixels": int(values.size),
        "word_metrics": word_rows,
        "minimum_word_survival": min(survivals) if survivals else 0.0,
        "minimum_word_delta_e76": min(contrasts) if contrasts else 0.0,
        "minimum_word_effective_contrast": min(effective) if effective else 0.0,
        "p10_word_effective_contrast": float(np.percentile(effective, 10)) if effective else 0.0,
        "mean_word_effective_contrast": float(np.mean(effective)) if effective else 0.0,
        "readability_gate_passed": gate,
    }


def candidate_rank_key(record: dict) -> tuple:
    """Preregistered detector-only selection key; larger is better."""

    legibility = record["legibility"]
    return (
        int(legibility["readability_gate_passed"]),
        float(legibility["minimum_word_effective_contrast"]),
        float(legibility["p10_word_effective_contrast"]),
        float(legibility["mean_word_effective_contrast"]),
        float(legibility["minimum_word_survival"]),
        -int(record["candidate_index"]),
    )


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    value = np.clip(rgb / 255.0, 0.0, 1.0)
    linear = np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)
    xyz = linear @ np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    ).T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > epsilon, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    return np.stack(
        [116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


def _delta_e76(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.linalg.norm(_srgb_to_lab(first) - _srgb_to_lab(second), axis=2)


def _word_metrics(
    *,
    image_size: tuple[int, int],
    layout_bbox: tuple[int, int, int, int] | list[int],
    attack_text: str,
    candidate_spec: CandidateSpecV4,
    blocked: np.ndarray,
    delta_e: np.ndarray,
) -> list[dict]:
    bbox = tuple(int(value) for value in layout_bbox)
    x0, y0, x1, y1 = bbox
    patch_width, patch_height = x1 - x0, y1 - y0
    font, _, left, top = _fit_single_line(attack_text, patch_width, patch_height)
    text_bbox = font.getbbox(attack_text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    line_x = (patch_width - text_width) / 2.0 - left
    if candidate_spec.vertical_anchor == "top":
        target_top = 3.0
    elif candidate_spec.vertical_anchor == "bottom":
        target_top = float(patch_height - text_height - 3)
    else:
        target_top = (patch_height - text_height) / 2.0
    line_y = target_top - top
    rows = []
    prefix = ""
    for word_index, word in enumerate(attack_text.split()):
        prefix_width = font.getlength(prefix)
        local = Image.new("L", (patch_width, patch_height), 0)
        draw = ImageDraw.Draw(local)
        word_x = line_x + prefix_width
        if candidate_spec.variant == "dual-chroma":
            draw.text((word_x - 1, line_y), word, font=font, fill=255)
            draw.text((word_x + 1, line_y + 1), word, font=font, fill=255)
        else:
            draw.text((word_x, line_y), word, font=font, fill=255)
        full = np.zeros((image_size[1], image_size[0]), dtype=bool)
        full[y0:y1, x0:x1] = np.asarray(local) > 0
        total = int(np.count_nonzero(full))
        remaining = full & ~blocked
        remaining_count = int(np.count_nonzero(remaining))
        survival = remaining_count / total if total else 0.0
        median_contrast = float(np.median(delta_e[remaining])) if remaining_count else 0.0
        rows.append({
            "word_index": word_index,
            "word": word,
            "carrier_pixels": total,
            "remaining_pixels": remaining_count,
            "survival_fraction": survival,
            "median_delta_e76": median_contrast,
            "effective_contrast": survival * survival * median_contrast,
        })
        prefix += word + " "
    return rows
