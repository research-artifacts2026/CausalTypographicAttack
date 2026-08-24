"""RIO-Bench protocol helpers without a hard dependency on HF datasets.

The official RIO repository remains the source of truth for final evaluation.
These helpers only materialize paired images, preserve attack metadata, and
provide an independently testable MC parser for per-query bookkeeping.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from typing import Iterable, Iterator

from .question_bench import normalize_answer, parse_multiple_choice_options


RIO_CONFIG_GROUPS = {
    "obj_mc": (
        "obj_clean__mc_clean",
        "obj_attack__mc_easy",
        "obj_attack__mc_medium",
        "obj_attack__mc_hard",
    ),
    "obj_oe": (
        "obj_clean__oe_clean",
        "obj_attack__oe_easy",
        "obj_attack__oe_medium",
        "obj_attack__oe_hard",
    ),
    "txt_oe": (
        "txt_clean__oe_clean",
        "txt_attack__oe_easy",
        "txt_attack__oe_hard",
    ),
}

RIO_CONDITION_BY_CONFIG = {
    "obj_clean__mc_clean": "no_attack",
    "obj_clean__oe_clean": "no_attack",
    "txt_clean__oe_clean": "no_attack",
    "obj_attack__mc_easy": "rio_typography_easy",
    "obj_attack__mc_medium": "rio_typography_medium",
    "obj_attack__mc_hard": "rio_typography_hard",
    "obj_attack__oe_easy": "rio_typography_easy",
    "obj_attack__oe_medium": "rio_typography_medium",
    "obj_attack__oe_hard": "rio_typography_hard",
    "txt_attack__oe_easy": "rio_text_attack_easy",
    "txt_attack__oe_hard": "rio_text_attack_hard",
}


def stable_reservoir(rows: Iterable[dict], limit: int, seed: int) -> list[dict]:
    """Choose the globally smallest stable hashes, independent of row order."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    ranked = []
    for row in rows:
        qid = str(row["question_id"])
        rank = hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest()
        ranked.append((rank, qid, row))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked[:limit]]


def index_unique(rows: Iterable[dict]) -> OrderedDict[str, dict]:
    indexed: OrderedDict[str, dict] = OrderedDict()
    for row in rows:
        qid = str(row["question_id"])
        if qid in indexed:
            raise ValueError(f"duplicate question_id: {qid}")
        indexed[qid] = row
    return indexed


def answer_letter(value: object, choices: object) -> str | None:
    parsed = parse_multiple_choice_options("", choices)
    text = str(value).strip().lower()
    if text in parsed:
        return text
    normalized = normalize_answer(value)
    for letter, content in parsed.items():
        if normalized == normalize_answer(content):
            return letter
    return None


def prediction_letter(prediction: str, choices: object) -> str | None:
    parsed = parse_multiple_choice_options("", choices)
    if not parsed:
        return None
    lower = prediction.lower().strip()
    strong = re.search(
        r"(?:answer|assistant|choice|option)\s*(?:is|:)?\s*\(?([a-d])\)?\b",
        lower,
    )
    if strong and strong.group(1) in parsed:
        return strong.group(1)
    bracketed = re.search(r"(?:^|\s|\()([a-d])(?:\)|\.|,|\s|$)", lower)
    if bracketed and bracketed.group(1) in parsed:
        return bracketed.group(1)
    normalized = normalize_answer(prediction)
    matches = [
        (len(normalize_answer(content)), letter)
        for letter, content in parsed.items()
        if normalize_answer(content)
        and (normalize_answer(content) in normalized or normalized in normalize_answer(content))
    ]
    if not matches:
        return None
    return max(matches)[1]


def rio_mc_score(prediction: str, answer: object, choices: object) -> float:
    """Match the official RIO evaluator's letter-first/text-fallback rule."""
    gold = answer_letter(answer, choices)
    pred = prediction_letter(prediction, choices)
    return float(gold is not None and pred == gold)


def target_letter_from_attack_word(row: dict, seed: int) -> tuple[str, str]:
    choices = parse_multiple_choice_options(str(row.get("question", "")), row.get("choices"))
    gold = answer_letter(row.get("answer"), choices)
    if not choices or gold is None:
        raise ValueError("RIO MC row requires choices and a resolvable gold answer")
    attack_word = normalize_answer(row.get("attack_word", ""))
    for letter, content in choices.items():
        if letter != gold and normalize_answer(content) == attack_word:
            return letter, "rio-attack-word-option"
    alternatives = sorted(letter for letter in choices if letter != gold)
    qid = str(row["question_id"])
    rank = hashlib.sha256(f"{seed}:{qid}:rio-target".encode()).hexdigest()
    return alternatives[int(rank[:16], 16) % len(alternatives)], "deterministic-rio-distractor"


def common_question_ids(indexes: Iterable[dict[str, dict]]) -> set[str]:
    sets = [set(index) for index in indexes]
    return set.intersection(*sets) if sets else set()

