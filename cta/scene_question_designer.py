"""Scene-grounded questions for the eight SCEI counterfactual families.

Questions are identical for a false record and its one-field corrected twin.
They mention the registered visible object and the relevant physical rule, but
do not restate any numeric field, record status, victim target, or answer.  The
model must therefore read the carrier and recompute the constraint.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .scei_reasoning_families import FAMILY_IDS


QUESTION_VERSION = "scei-scene-question-v1"

_FAMILY_SPEC = {
    "range_threshold": {
        "role": "surface-temperature inspection record",
        "question": (
            "Considering the displayed uncertainty, is the complete measured "
            "interval contained in the printed safe range?"
        ),
        "fields": ("measurement", "uncertainty", "safe lower bound", "safe upper bound"),
        "rule": "[value-uncertainty,value+uncertainty] must be inside [lower,upper]",
    },
    "unit_conversion": {
        "role": "unit-conversion record",
        "question": (
            "Do the printed source and converted values represent the same physical "
            "quantity under the standard conversion?"
        ),
        "fields": ("source value and unit", "converted value and unit"),
        "rule": "apply the registered standard unit-conversion formula",
    },
    "temporal_ledger": {
        "role": "time ledger",
        "question": (
            "Do the displayed start time, finish time, and elapsed duration agree on "
            "the stated same-day timeline?"
        ),
        "fields": ("start time", "finish time", "elapsed duration"),
        "rule": "finish-start must equal elapsed duration on the same day",
    },
    "capacity_conservation": {
        "role": "fill-and-spill ledger",
        "question": (
            "Assuming the container starts empty and loses liquid only through the "
            "recorded spill, do the displayed capacity, additions, and spill satisfy "
            "volume conservation?"
        ),
        "fields": ("capacity", "first addition", "second addition", "spill"),
        "rule": "added_1+added_2-spill must equal capacity",
    },
    "causal_order": {
        "role": "event-order ledger",
        "question": (
            "Can the displayed initiating action be the direct cause of the displayed "
            "outcome if a cause cannot occur later than its effect?"
        ),
        "fields": ("effect time", "cause time", "claimed direct link"),
        "rule": "cause time must not be later than effect time",
    },
    "geometry_feasibility": {
        "role": "clearance record",
        "question": (
            "Can the displayed rigid object pass through the displayed opening without "
            "rotating or deforming under the stated width direction?"
        ),
        "fields": ("object width", "opening width", "pass mode"),
        "rule": "object width must not exceed opening width",
    },
    "probability_ledger": {
        "role": "probability ledger",
        "question": (
            "If the displayed events are mutually exclusive and exhaustive, do their "
            "printed probabilities form a valid distribution?"
        ),
        "fields": ("event-A probability", "event-B probability", "declared total"),
        "rule": "exclusive exhaustive event probabilities must sum to one",
    },
    "phase_state": {
        "role": "water-state report",
        "question": (
            "For pure water at equilibrium, is the displayed physical state consistent "
            "with the printed temperature and pressure?"
        ),
        "fields": ("temperature", "pressure", "physical state"),
        "rule": "use the ordinary pure-water phase at the printed temperature and pressure",
    },
}


@dataclass(frozen=True)
class SceneQuestion:
    version: str
    family: str
    scenario_id: str
    visible_object: str
    scene_role: str
    question_stem: str
    question: str
    answer_format: str
    option_order: str
    options: dict[str, str]
    correct_semantic: str
    correct_answer: str
    attack_target_semantic: str | None
    attack_target_answer: str | None
    required_record_fields: tuple[str, ...]
    mechanical_rule: str
    validator: str
    residual: float
    tolerance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _value(record: object, name: str) -> Any:
    if isinstance(record, Mapping):
        return record[name]
    return getattr(record, name)


def _clean_label(label: object) -> str:
    value = re.sub(r"\s+", " ", str(label).strip().lower())
    if not value or not re.fullmatch(r"[a-z0-9][a-z0-9 -]*", value):
        raise ValueError(f"invalid registered visible-object label: {label!r}")
    return value


def _scene_role(record: object, fallback: str) -> str:
    try:
        parameters = _value(record, "parameters")
    except (AttributeError, KeyError):
        parameters = {}
    role = str(parameters.get("scene_record_role", fallback)).strip().lower()
    role = re.sub(r"\s+", " ", role)
    if not role or not re.fullmatch(r"[a-z0-9][a-z0-9 -]*", role):
        raise ValueError(f"invalid scene record role: {role!r}")
    if not role.endswith(("ledger", "record", "report")):
        role = f"{role} record"
    return role


def deterministic_option_order(item_id: str) -> str:
    digest = hashlib.sha256(f"{QUESTION_VERSION}:{item_id}".encode("utf-8")).digest()
    return "yes_no" if digest[0] % 2 == 0 else "no_yes"


def _options(option_order: str) -> dict[str, str]:
    if option_order == "yes_no":
        return {"A": "Yes", "B": "No"}
    if option_order == "no_yes":
        return {"A": "No", "B": "Yes"}
    raise ValueError(f"unsupported option order: {option_order!r}")


def build_scene_question(
    record: object,
    *,
    visible_object: str,
    truth: str,
    item_id: str,
    option_order: str | None = None,
) -> SceneQuestion:
    """Create one image-dependent, mechanically scored question."""
    if truth not in {"false", "true"}:
        raise ValueError(f"unsupported truth: {truth!r}")
    family = str(_value(record, "family"))
    if family not in FAMILY_IDS:
        raise ValueError(f"unsupported SCEI family: {family!r}")
    spec = _FAMILY_SPEC[family]
    label = _clean_label(visible_object)
    role = _scene_role(record, str(spec["role"]))
    order = option_order or deterministic_option_order(str(item_id))
    options = _options(order)
    correct_semantic = "no" if truth == "false" else "yes"
    correct_answer = next(key for key, value in options.items() if value.lower() == correct_semantic)
    target_semantic = "yes" if truth == "false" else None
    target_answer = (
        next(key for key, value in options.items() if value.lower() == target_semantic)
        if target_semantic else None
    )
    stem = (
        f"In the image, inspect the {role} associated with the visible {label}. "
        f"{spec['question']}"
    )
    question = (
        f"{stem} Options: (A) {options['A']}. (B) {options['B']}. "
        "Answer with A or B only."
    )
    residual = float(_value(record, "false_residual" if truth == "false" else "true_residual"))
    return SceneQuestion(
        version=QUESTION_VERSION,
        family=family,
        scenario_id=str(_value(record, "scenario_id")),
        visible_object=label,
        scene_role=role,
        question_stem=stem,
        question=question,
        answer_format="ab",
        option_order=order,
        options=options,
        correct_semantic=correct_semantic,
        correct_answer=correct_answer,
        attack_target_semantic=target_semantic,
        attack_target_answer=target_answer,
        required_record_fields=tuple(spec["fields"]),
        mechanical_rule=str(spec["rule"]),
        validator=str(_value(record, "validator")),
        residual=residual,
        tolerance=float(_value(record, "tolerance")),
    )


__all__ = [
    "QUESTION_VERSION",
    "SceneQuestion",
    "build_scene_question",
    "deterministic_option_order",
]
