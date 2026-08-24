"""Deterministic camera-like transforms for simulated robustness tests.

These outputs are deliberately named simulated captures. They are not a
substitute for photographing printed attacks in a real physical environment.
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class CaptureProfile:
    name: str
    perspective_fraction: float
    brightness: float
    blur_radius: float
    downscale: float
    jpeg_quality: int


PROFILES = {
    "mild": CaptureProfile("mild", 0.025, 0.92, 0.55, 0.78, 85),
    "medium": CaptureProfile("medium", 0.060, 0.76, 1.05, 0.55, 65),
    "severe": CaptureProfile("severe", 0.105, 0.60, 1.65, 0.36, 45),
}


def _perspective_coefficients(source, destination):
    matrix = []
    vector = []
    for (x, y), (u, v) in zip(destination, source):
        matrix.extend(([x, y, 1, 0, 0, 0, -u * x, -u * y],
                       [0, 0, 0, x, y, 1, -v * x, -v * y]))
        vector.extend((u, v))
    return np.linalg.solve(np.asarray(matrix), np.asarray(vector))


def _corners(width: int, height: int, fraction: float, seed: int):
    digest = hashlib.sha256(str(seed).encode()).digest()
    signs = [1 if byte % 2 else -1 for byte in digest[:8]]
    dx, dy = width * fraction, height * fraction
    base = [(0, 0), (width, 0), (width, height), (0, height)]
    moved = []
    for index, (x, y) in enumerate(base):
        mx = min(width, max(0, x + signs[index * 2] * dx))
        my = min(height, max(0, y + signs[index * 2 + 1] * dy))
        moved.append((mx, my))
    return base, moved


def simulate_capture(
    source: str | Path, output: str | Path, profile: CaptureProfile, seed: int,
) -> dict:
    image = Image.open(source).convert("RGB")
    original_size = image.size
    source_corners, destination = _corners(image.width, image.height, profile.perspective_fraction, seed)
    coeffs = _perspective_coefficients(source_corners, destination)
    image = image.transform(
        image.size, Image.Transform.PERSPECTIVE, coeffs,
        resample=Image.Resampling.BICUBIC, fillcolor=(32, 32, 32),
    )
    image = ImageEnhance.Brightness(image).enhance(profile.brightness)
    image = image.filter(ImageFilter.GaussianBlur(profile.blur_radius))
    small = (
        max(96, round(image.width * profile.downscale)),
        max(96, round(image.height * profile.downscale)),
    )
    image = image.resize(small, Image.Resampling.LANCZOS).resize(original_size, Image.Resampling.BICUBIC)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=profile.jpeg_quality, optimize=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(buffer.getvalue())
    return {
        "profile": asdict(profile),
        "seed": seed,
        "original_size": list(original_size),
        "downscaled_size": list(small),
        "destination_corners": [[round(x, 3), round(y, 3)] for x, y in destination],
    }
