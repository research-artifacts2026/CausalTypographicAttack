from __future__ import annotations

import re
from pathlib import Path

from .render import RenderedAttack, mask_bbox


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[a-zA-Z]+", text) if len(t) > 2}


def consistency_defense(artifact: RenderedAttack, scene: dict, target_label: str, output: Path) -> tuple[str, dict]:
    """Lightweight SAGE-style wrapper: mask text that conflicts with visible-object tokens.

    It intentionally checks scene-text compatibility only, not physical possibility. This is
    a transparent proxy, not an implementation of SAGE.
    """
    visible = _tokens(target_label)
    for obj in scene.get("objects", []) if isinstance(scene.get("objects", []), list) else []:
        visible |= _tokens(str(obj))
    text_tokens = _tokens(artifact.text)
    wrong_tokens = _tokens(artifact.target_wrong_label or "")
    contradictory = bool(wrong_tokens) and bool(text_tokens & wrong_tokens) and not bool(visible & wrong_tokens)
    if contradictory:
        defended = mask_bbox(artifact.image_path, artifact.bbox, output)
        action = "mask"
    else:
        defended = artifact.image_path
        action = "pass"
    return defended, {
        "defense": "consistency",
        "action": action,
        "scene_tokens": sorted(visible),
        "note": "lightweight lexical scene-text consistency proxy; not SAGE",
    }


def ocr_mask_defense(artifact: RenderedAttack, output: Path) -> tuple[str, dict]:
    defended = mask_bbox(artifact.image_path, artifact.bbox, output)
    return defended, {
        "defense": "ocr_mask",
        "action": "mask",
        "backend": "renderer_bbox",
        "note": "oracle text-region upper bound; replace with OCR detector for deployment",
    }

