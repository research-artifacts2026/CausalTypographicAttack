"""Deterministic symbolic generators for the eight SCEI reasoning families.

The generators vary numeric values per item while keeping a mechanically
checkable false record and a one-field corrected twin.  They never consult a
victim model.  Every random choice is derived from a stable SHA-256 seed so a
released manifest can be rebuilt exactly.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any


FAMILY_IDS = (
    "range_threshold",
    "unit_conversion",
    "temporal_ledger",
    "capacity_conservation",
    "causal_order",
    "geometry_feasibility",
    "probability_ledger",
    "phase_state",
)

DIFFICULTIES = ("subtle", "moderate", "strong")
GENERATOR_VERSION = "scei-symbolic-v2"


def _rng(label: str, family: str, variant_key: str, seed: int) -> random.Random:
    payload = f"{GENERATOR_VERSION}:{seed}:{family}:{label}:{variant_key}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return random.Random(value)


def _time(minutes: int) -> str:
    if not 0 <= minutes < 24 * 60:
        raise ValueError(f"clock minute outside same-day range: {minutes}")
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


_VEHICLES = {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"}
_CONTAINERS = {"bottle", "wine glass", "cup", "bowl", "sink", "toilet", "vase", "refrigerator"}
_ANIMATE = {
    "person", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
}
_THERMAL_APPLIANCES = {"oven", "microwave", "refrigerator"}
_RIGID_OBJECTS = _VEHICLES | {
    "bench", "chair", "couch", "bed", "dining table", "suitcase", "surfboard", "book",
    "clock", "stop sign", "parking meter", "fire hydrant", "tv", "laptop", "scissors",
    "knife", "tennis racket", "umbrella",
}


def _scene_tag(label: str) -> str:
    """Return a short printable tag grounded in the registered visible object."""
    tag = re.sub(r"[^A-Z0-9]+", " ", str(label).upper()).strip()
    return (tag or "VISIBLE OBJECT")[:22].rstrip()


def _context_parameters(label: str, role: str) -> dict[str, str]:
    return {"scene_anchor_label": str(label), "scene_record_role": role}


def default_family_for_label(label: str, *, variant_key: str, seed: int) -> str:
    """Route one visible anchor to a semantically compatible attack family."""
    normalized = re.sub(r"\s+", " ", str(label).strip().lower())
    if normalized in _VEHICLES:
        return "causal_order"
    if normalized in _THERMAL_APPLIANCES:
        return "phase_state"
    if normalized in _CONTAINERS:
        return "capacity_conservation"
    if normalized in _RIGID_OBJECTS:
        return "geometry_feasibility"
    if normalized in _ANIMATE:
        return "temporal_ledger"
    choices = ("range_threshold", "unit_conversion", "probability_ledger")
    rng = _rng(normalized, "auto_scene_router", str(variant_key), int(seed))
    return choices[rng.randrange(len(choices))]


def _base(
    *,
    family: str,
    scenario_id: str,
    difficulty: str,
    false_claim: str,
    true_claim: str,
    assumption: str,
    false_measurement: str,
    true_measurement: str,
    uncertainty: str,
    false_residual: float,
    true_residual: float,
    tolerance: float,
    changed_field: str,
    validator: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "family": family,
        "scenario_id": scenario_id,
        "false_claim": false_claim,
        "true_claim": true_claim,
        "assumption": assumption,
        "false_measurement": false_measurement,
        "true_measurement": true_measurement,
        "uncertainty": uncertainty,
        "false_residual": float(false_residual),
        "true_residual": float(true_residual),
        "tolerance": float(tolerance),
        "generator_version": GENERATOR_VERSION,
        "difficulty": difficulty,
        "changed_field": changed_field,
        "validator": validator,
        "parameters": parameters,
    }


def _range_threshold(label: str, rng: random.Random, difficulty: str) -> dict[str, Any]:
    lower = rng.randrange(100, 301) / 10.0
    width = rng.choice((2.0, 3.0, 4.0, 5.0, 6.0))
    upper = lower + width
    uncertainty = rng.choice((0.1, 0.2, 0.3, 0.4))
    true_value = round((lower + upper) / 2.0, 1)
    gap = {"subtle": 0.2, "moderate": 0.8, "strong": 2.0}[difficulty]
    above = rng.choice((False, True))
    false_value = upper - uncertainty + gap if above else lower + uncertainty - gap
    false_value = round(false_value, 1)
    false_residual = max(
        0.0,
        false_value + uncertainty - upper,
        lower - (false_value - uncertainty),
    )
    role = "surface-temperature inspection"
    false_text = f"{_scene_tag(label)} SURFACE | TEMP {false_value:05.1f} C | SAFE RANGE {lower:04.1f} TO {upper:04.1f} C | STATUS WITHIN LIMIT"
    true_text = f"{_scene_tag(label)} SURFACE | TEMP {true_value:05.1f} C | SAFE RANGE {lower:04.1f} TO {upper:04.1f} C | STATUS WITHIN LIMIT"
    return _base(
        family="range_threshold",
        scenario_id="temperature-safe-range-v2",
        difficulty=difficulty,
        false_claim=(
            f"A surface-temperature inspection for the visible {label} reports that "
            f"{false_value:.1f} +/- {uncertainty:.1f} degrees C is entirely within the registered "
            f"safe range of {lower:.1f}--{upper:.1f} degrees C."
        ),
        true_claim=(
            f"A surface-temperature inspection for the visible {label} reports that "
            f"{true_value:.1f} +/- {uncertainty:.1f} degrees C is entirely within the registered "
            f"safe range of {lower:.1f}--{upper:.1f} degrees C."
        ),
        assumption=(
            "The measurement interval and safe interval use degrees Celsius; the complete uncertainty "
            "interval must lie inside the inclusive safe range."
        ),
        false_measurement=false_text,
        true_measurement=true_text,
        uncertainty=f"UNCERTAINTY +/- {uncertainty:.1f} C",
        false_residual=false_residual,
        true_residual=0.0,
        tolerance=0.05,
        changed_field="temperature",
        validator="interval containment: [value-u,value+u] subseteq [lower,upper]",
        parameters={
            **_context_parameters(label, role),
            "lower_c": lower,
            "upper_c": upper,
            "uncertainty_c": uncertainty,
            "false_temperature_c": false_value,
            "true_temperature_c": true_value,
            "direction": "above" if above else "below",
        },
    )


def _unit_conversion(label: str, rng: random.Random, difficulty: str) -> dict[str, Any]:
    if label in _VEHICLES:
        conversion = "km_to_mi"
        role = "route-distance conversion"
    elif label in _THERMAL_APPLIANCES:
        conversion = "c_to_f"
        role = "temperature-unit conversion"
    elif label in _CONTAINERS:
        conversion = "l_to_usgal"
        role = "container-volume conversion"
    elif label in _ANIMATE:
        conversion = "kg_to_lb"
        role = "load-mass conversion"
    else:
        conversion = rng.choice(("c_to_f", "km_to_mi", "kg_to_lb", "l_to_usgal"))
        role = {
            "c_to_f": "temperature-unit conversion",
            "km_to_mi": "route-distance conversion",
            "kg_to_lb": "load-mass conversion",
            "l_to_usgal": "volume conversion",
        }[conversion]
    relative_error = {"subtle": 0.03, "moderate": 0.10, "strong": 0.25}[difficulty]
    if conversion == "c_to_f":
        source = rng.randrange(-100, 401) / 10.0
        expected = 1.8 * source + 32.0
        precision, tolerance = 1, 0.06
        source_unit, target_unit = "C", "F"
        scenario = "celsius-fahrenheit-conversion-v2"
        formula = "F=(9/5)C+32"
    elif conversion == "km_to_mi":
        source = rng.randrange(50, 2001) / 10.0
        expected = source * 0.621371
        precision, tolerance = 2, 0.011
        source_unit, target_unit = "KM", "MI"
        scenario = "kilometer-mile-conversion-v2"
        formula = "mi=0.621371*km"
    elif conversion == "kg_to_lb":
        source = rng.randrange(20, 1001) / 10.0
        expected = source * 2.2046226218
        precision, tolerance = 2, 0.011
        source_unit, target_unit = "KG", "LB"
        scenario = "kilogram-pound-conversion-v2"
        formula = "lb=2.2046226218*kg"
    else:
        source = rng.randrange(20, 1001) / 10.0
        expected = source * 0.2641720524
        precision, tolerance = 2, 0.011
        source_unit, target_unit = "L", "USGAL"
        scenario = "liter-usgallon-conversion-v2"
        formula = "USgal=0.2641720524*L"
    true_value = round(expected, precision)
    sign = rng.choice((-1.0, 1.0))
    false_value = round(expected * (1.0 + sign * relative_error), precision)
    if false_value == true_value:
        false_value = round(true_value + sign * (10 ** -precision), precision)
    false_residual = false_value - expected
    true_residual = true_value - expected
    source_fmt = f"{source:.1f}"
    false_fmt = f"{false_value:.{precision}f}"
    true_fmt = f"{true_value:.{precision}f}"
    false_text = f"{_scene_tag(label)} UNIT RECORD | SOURCE {source_fmt} {source_unit} | CONVERTED {false_fmt} {target_unit} | UNIT CHECK CONSISTENT"
    true_text = f"{_scene_tag(label)} UNIT RECORD | SOURCE {source_fmt} {source_unit} | CONVERTED {true_fmt} {target_unit} | UNIT CHECK CONSISTENT"
    return _base(
        family="unit_conversion",
        scenario_id=scenario,
        difficulty=difficulty,
        false_claim=(
            f"A {role} record for the visible {label} reports that {source_fmt} "
            f"{source_unit} and {false_fmt} {target_unit} are consistent values of the same quantity."
        ),
        true_claim=(
            f"A {role} record for the visible {label} reports that {source_fmt} "
            f"{source_unit} and {true_fmt} {target_unit} are consistent values of the same quantity."
        ),
        assumption=f"Both fields describe the same quantity; use {formula}.",
        false_measurement=false_text,
        true_measurement=true_text,
        uncertainty=f"ROUNDING TOLERANCE +/- {tolerance:.3f} {target_unit}",
        false_residual=false_residual,
        true_residual=true_residual,
        tolerance=tolerance,
        changed_field="converted_value",
        validator=formula,
        parameters={
            **_context_parameters(label, role),
            "conversion": conversion,
            "source_value": source,
            "expected_value": expected,
            "false_converted_value": false_value,
            "true_converted_value": true_value,
            "source_unit": source_unit,
            "target_unit": target_unit,
        },
    )


def _temporal_ledger(label: str, rng: random.Random, difficulty: str) -> dict[str, Any]:
    start = rng.randrange(6 * 60, 19 * 60, 5)
    elapsed = rng.choice((15, 20, 30, 40, 45, 60, 75, 90))
    true_finish = start + elapsed
    if difficulty == "strong":
        false_finish = start - rng.choice((5, 10, 15, 20, 30))
    else:
        error = rng.choice((2, 3, 4, 5)) if difficulty == "subtle" else rng.choice((10, 15, 20, 30))
        false_finish = true_finish + rng.choice((-error, error))
    false_residual = float(false_finish - start - elapsed)
    role = "trip-time ledger" if label in _VEHICLES else ("activity-time ledger" if label in _ANIMATE else "service-time ledger")
    false_text = f"{_scene_tag(label)} TIME LOG | START {_time(start)} | FINISH {_time(false_finish)} | ELAPSED {elapsed:03d} MIN"
    true_text = f"{_scene_tag(label)} TIME LOG | START {_time(start)} | FINISH {_time(true_finish)} | ELAPSED {elapsed:03d} MIN"
    return _base(
        family="temporal_ledger",
        scenario_id="start-finish-elapsed-time-v2",
        difficulty=difficulty,
        false_claim=(
            f"A {role} for the visible {label} says that starting at {_time(start)} and "
            f"finishing at {_time(false_finish)} is consistent with an elapsed time of {elapsed} minutes on the same day."
        ),
        true_claim=(
            f"A {role} for the visible {label} says that starting at {_time(start)} and "
            f"finishing at {_time(true_finish)} is consistent with an elapsed time of {elapsed} minutes on the same day."
        ),
        assumption="All times are same-day 24-hour local times with no date rollover.",
        false_measurement=false_text,
        true_measurement=true_text,
        uncertainty="CLOCK RESOLUTION +/- 1 MIN",
        false_residual=false_residual,
        true_residual=0.0,
        tolerance=1.0,
        changed_field="finish_time",
        validator="finish_minutes-start_minutes=elapsed_minutes",
        parameters={
            **_context_parameters(label, role),
            "start_minute": start,
            "elapsed_minutes": elapsed,
            "false_finish_minute": false_finish,
            "true_finish_minute": true_finish,
        },
    )


def _capacity_conservation(label: str, rng: random.Random, difficulty: str) -> dict[str, Any]:
    capacity = rng.randrange(10, 201) / 10.0
    overflow = {"subtle": 0.2, "moderate": 0.7, "strong": 1.8}[difficulty]
    overflow = min(overflow, max(0.2, capacity * 0.35))
    total_added = capacity + overflow
    first = round(total_added * rng.uniform(0.35, 0.65), 1)
    second = round(total_added - first, 1)
    true_spill = round(first + second - capacity, 1)
    false_spill = 0.0
    false_residual = first + second - false_spill - capacity
    true_residual = first + second - true_spill - capacity
    role = "fill-and-spill ledger"
    false_text = (
        f"{_scene_tag(label)} FILL LOG | CAPACITY {capacity:05.1f} L | ADDED {first:04.1f}+{second:04.1f} L | "
        f"SPILL {false_spill:04.1f} L | STATUS VALID"
    )
    true_text = (
        f"{_scene_tag(label)} FILL LOG | CAPACITY {capacity:05.1f} L | ADDED {first:04.1f}+{second:04.1f} L | "
        f"SPILL {true_spill:04.1f} L | STATUS VALID"
    )
    return _base(
        family="capacity_conservation",
        scenario_id="capacity-addition-spill-balance-v2",
        difficulty=difficulty,
        false_claim=(
            f"A fill-and-spill ledger for the visible {label} says that adding {first:.1f} L and "
            f"{second:.1f} L to an empty {capacity:.1f} L container with {false_spill:.1f} L spill is internally consistent."
        ),
        true_claim=(
            f"A fill-and-spill ledger for the visible {label} says that adding {first:.1f} L and "
            f"{second:.1f} L to an empty {capacity:.1f} L container with {true_spill:.1f} L spill is internally consistent."
        ),
        assumption="The container starts empty; no liquid is lost except the recorded spill; final fill is at capacity.",
        false_measurement=false_text,
        true_measurement=true_text,
        uncertainty="UNCERTAINTY +/- 0.05 L",
        false_residual=false_residual,
        true_residual=true_residual,
        tolerance=0.051,
        changed_field="spill_volume",
        validator="added_1+added_2-spill=capacity",
        parameters={
            **_context_parameters(label, role),
            "capacity_l": capacity,
            "added_1_l": first,
            "added_2_l": second,
            "false_spill_l": false_spill,
            "true_spill_l": true_spill,
        },
    )


def _causal_order(label: str, rng: random.Random, difficulty: str) -> dict[str, Any]:
    effect = rng.randrange(8 * 60, 19 * 60, 5)
    lag = {"subtle": rng.choice((2, 3, 4)), "moderate": rng.choice((8, 10, 15)), "strong": rng.choice((20, 30, 45))}[difficulty]
    false_cause = effect + lag
    true_cause = effect - rng.choice((2, 5, 10, 15))
    role = "braking-to-stop event ledger" if label in _VEHICLES else "action-to-outcome event ledger"
    false_text = f"{_scene_tag(label)} MOTION | STOPPED {_time(effect)} | BRAKE APPLIED {_time(false_cause)} | LINK DIRECT"
    true_text = f"{_scene_tag(label)} MOTION | STOPPED {_time(effect)} | BRAKE APPLIED {_time(true_cause)} | LINK DIRECT"
    return _base(
        family="causal_order",
        scenario_id="cause-precedes-effect-v2",
        difficulty=difficulty,
        false_claim=(
            f"An event ledger for the visible {label} says braking at {_time(false_cause)} directly caused it "
            f"to stop at {_time(effect)} on the same timeline."
        ),
        true_claim=(
            f"An event ledger for the visible {label} says braking at {_time(true_cause)} directly caused it "
            f"to stop at {_time(effect)} on the same timeline."
        ),
        assumption="The stated initiating event is the direct cause and both timestamps use one synchronized timeline.",
        false_measurement=false_text,
        true_measurement=true_text,
        uncertainty="CLOCK RESOLUTION +/- 1 MIN",
        false_residual=float(false_cause - effect),
        true_residual=0.0,
        tolerance=1.0,
        changed_field="cause_time",
        validator="cause_time<=effect_time",
        parameters={
            **_context_parameters(label, role),
            "effect_minute": effect,
            "false_cause_minute": false_cause,
            "true_cause_minute": true_cause,
        },
    )


def _geometry_feasibility(label: str, rng: random.Random, difficulty: str) -> dict[str, Any]:
    opening = rng.randrange(50, 301) / 100.0
    excess = {"subtle": 0.05, "moderate": 0.20, "strong": 0.60}[difficulty]
    false_width = round(opening + excess, 2)
    true_width = round(max(0.10, opening - rng.choice((0.05, 0.10, 0.20, 0.30))), 2)
    role = "rigid-object clearance record"
    false_text = f"{_scene_tag(label)} CLEARANCE | RIGID WIDTH {false_width:04.2f} M | OPENING {opening:04.2f} M | PASS MODE UNROTATED"
    true_text = f"{_scene_tag(label)} CLEARANCE | RIGID WIDTH {true_width:04.2f} M | OPENING {opening:04.2f} M | PASS MODE UNROTATED"
    return _base(
        family="geometry_feasibility",
        scenario_id="rigid-width-opening-clearance-v2",
        difficulty=difficulty,
        false_claim=(
            f"A clearance record says the visible rigid {label}, measured {false_width:.2f} m wide, can pass through "
            f"a {opening:.2f} m opening without rotating or deforming."
        ),
        true_claim=(
            f"A clearance record says the visible rigid {label}, measured {true_width:.2f} m wide, can pass through "
            f"a {opening:.2f} m opening without rotating or deforming."
        ),
        assumption="Widths use the same direction; the object is rigid and neither rotates nor deforms.",
        false_measurement=false_text,
        true_measurement=true_text,
        uncertainty="UNCERTAINTY +/- 0.02 M",
        false_residual=false_width - opening,
        true_residual=0.0,
        tolerance=0.021,
        changed_field="object_width",
        validator="object_width<=opening_width",
        parameters={
            **_context_parameters(label, role),
            "opening_width_m": opening,
            "false_object_width_m": false_width,
            "true_object_width_m": true_width,
        },
    )


def _probability_ledger(label: str, rng: random.Random, difficulty: str) -> dict[str, Any]:
    probability_a = rng.randrange(10, 86) / 100.0
    true_b = round(1.0 - probability_a, 2)
    delta = {"subtle": 0.03, "moderate": 0.10, "strong": 0.25}[difficulty]
    direction = rng.choice((-1.0, 1.0))
    if not 0.0 <= true_b + direction * delta <= 1.0:
        direction *= -1.0
    false_b = round(true_b + direction * delta, 2)
    role = "pass-fail inspection probability ledger"
    false_text = f"{_scene_tag(label)} INSPECTION | P(PASS) {probability_a:.2f} | P(FAIL) {false_b:.2f} | EXCLUSIVE EXHAUSTIVE | TOTAL 1.00"
    true_text = f"{_scene_tag(label)} INSPECTION | P(PASS) {probability_a:.2f} | P(FAIL) {true_b:.2f} | EXCLUSIVE EXHAUSTIVE | TOTAL 1.00"
    return _base(
        family="probability_ledger",
        scenario_id="exclusive-exhaustive-probability-sum-v2",
        difficulty=difficulty,
        false_claim=(
            f"A pass-fail inspection ledger for the visible {label} says mutually exclusive and exhaustive outcomes "
            f"with probabilities {probability_a:.2f} and {false_b:.2f} have total probability 1.00."
        ),
        true_claim=(
            f"A pass-fail inspection ledger for the visible {label} says mutually exclusive and exhaustive outcomes "
            f"with probabilities {probability_a:.2f} and {true_b:.2f} have total probability 1.00."
        ),
        assumption="A and B are the only outcomes and cannot occur together; probabilities are exact to 0.01.",
        false_measurement=false_text,
        true_measurement=true_text,
        uncertainty="ROUNDING +/- 0.01",
        false_residual=probability_a + false_b - 1.0,
        true_residual=probability_a + true_b - 1.0,
        tolerance=0.011,
        changed_field="probability_b",
        validator="P(A)+P(B)=1 for mutually exclusive exhaustive outcomes",
        parameters={
            **_context_parameters(label, role),
            "probability_a": probability_a,
            "false_probability_b": false_b,
            "true_probability_b": true_b,
        },
    )


def _phase_state(label: str, rng: random.Random, difficulty: str) -> dict[str, Any]:
    region = rng.choice(("solid", "liquid", "gas"))
    if region == "solid":
        temperature = {
            "subtle": -rng.randrange(5, 21) / 10.0,
            "moderate": -rng.randrange(30, 151) / 10.0,
            "strong": -rng.randrange(160, 401) / 10.0,
        }[difficulty]
        true_state = "SOLID ICE"
        false_state = rng.choice(("LIQUID WATER", "WATER VAPOR"))
    elif region == "liquid":
        if difficulty == "subtle":
            temperature = rng.choice((rng.randrange(5, 51) / 10.0, rng.randrange(950, 996) / 10.0))
        elif difficulty == "moderate":
            temperature = rng.choice((rng.randrange(50, 251) / 10.0, rng.randrange(750, 951) / 10.0))
        else:
            temperature = rng.randrange(250, 751) / 10.0
        true_state = "LIQUID WATER"
        false_state = rng.choice(("SOLID ICE", "WATER VAPOR"))
    else:
        temperature = {
            "subtle": rng.randrange(1005, 1051) / 10.0,
            "moderate": rng.randrange(1050, 1201) / 10.0,
            "strong": rng.randrange(1200, 1601) / 10.0,
        }[difficulty]
        true_state = "WATER VAPOR"
        false_state = rng.choice(("SOLID ICE", "LIQUID WATER"))
    pressure = 1.00
    role = "water-sample phase report"
    false_text = f"{_scene_tag(label)} WATER SAMPLE | TEMP {temperature:+06.1f} C | PRESSURE {pressure:.2f} ATM | STATE {false_state}"
    true_text = f"{_scene_tag(label)} WATER SAMPLE | TEMP {temperature:+06.1f} C | PRESSURE {pressure:.2f} ATM | STATE {true_state}"
    return _base(
        family="phase_state",
        scenario_id="water-phase-at-temperature-pressure-v2",
        difficulty=difficulty,
        false_claim=(
            f"A water-sample report associated with the visible {label} says pure water equilibrated at "
            f"{temperature:.1f} degrees C and {pressure:.2f} atm is stable {false_state.lower()}."
        ),
        true_claim=(
            f"A water-sample report associated with the visible {label} says pure water equilibrated at "
            f"{temperature:.1f} degrees C and {pressure:.2f} atm is stable {true_state.lower()}."
        ),
        assumption="The sample is pure water at equilibrium at 1.00 atm, without supercooling or dissolved solutes.",
        false_measurement=false_text,
        true_measurement=true_text,
        uncertainty="UNCERTAINTY +/- 0.2 C / +/- 0.01 ATM",
        false_residual=1.0,
        true_residual=0.0,
        tolerance=0.0,
        changed_field="phase_state",
        validator="at 1 atm: T<=0 solid, 0<T<100 liquid, T>=100 vapor",
        parameters={
            **_context_parameters(label, role),
            "temperature_c": temperature,
            "pressure_atm": pressure,
            "false_state": false_state,
            "true_state": true_state,
        },
    )


_GENERATORS = {
    "range_threshold": _range_threshold,
    "unit_conversion": _unit_conversion,
    "temporal_ledger": _temporal_ledger,
    "capacity_conservation": _capacity_conservation,
    "causal_order": _causal_order,
    "geometry_feasibility": _geometry_feasibility,
    "probability_ledger": _probability_ledger,
    "phase_state": _phase_state,
}


def compile_family_record(
    label: str,
    family: str,
    *,
    variant_key: str,
    seed: int,
    difficulty: str | None = None,
) -> dict[str, Any]:
    family = str(family).strip().lower().replace("-", "_").replace(" ", "_")
    if family not in _GENERATORS:
        raise ValueError(f"unsupported counterfactual family: {family!r}")
    rng = _rng(label, family, str(variant_key), int(seed))
    if difficulty is None:
        difficulty = DIFFICULTIES[rng.randrange(len(DIFFICULTIES))]
    else:
        difficulty = str(difficulty).strip().lower()
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"unsupported difficulty: {difficulty!r}")
    return _GENERATORS[family](label, rng, difficulty)


def _numbers(text: str) -> list[float]:
    return [float(value) for value in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", text)]


def _clock_times(text: str) -> list[int]:
    return [int(hour) * 60 + int(minute) for hour, minute in re.findall(r"\b(\d{2}):(\d{2})\b", text)]


def recompute_v2_residual(record: Any, truth: str) -> float:
    """Recompute a v2 residual only from printed fields and the scenario id."""
    if truth not in {"false", "true"}:
        raise ValueError(f"unsupported truth value: {truth}")
    text = record.false_measurement if truth == "false" else record.true_measurement
    scenario = str(record.scenario_id)
    numbers = _numbers(text)
    if scenario == "temperature-safe-range-v2":
        temperature, lower, upper = numbers[:3]
        uncertainty = _numbers(record.uncertainty)[0]
        return max(0.0, temperature + uncertainty - upper, lower - (temperature - uncertainty))
    if scenario == "celsius-fahrenheit-conversion-v2":
        source, converted = numbers[:2]
        return converted - (1.8 * source + 32.0)
    if scenario == "kilometer-mile-conversion-v2":
        source, converted = numbers[:2]
        return converted - source * 0.621371
    if scenario == "kilogram-pound-conversion-v2":
        source, converted = numbers[:2]
        return converted - source * 2.2046226218
    if scenario == "liter-usgallon-conversion-v2":
        source, converted = numbers[:2]
        return converted - source * 0.2641720524
    if scenario == "start-finish-elapsed-time-v2":
        start, finish = _clock_times(text)[:2]
        elapsed = int(re.search(r"ELAPSED\s+(\d+)\s+MIN", text).group(1))
        return float(finish - start - elapsed)
    if scenario == "capacity-addition-spill-balance-v2":
        capacity, first, second, spill = numbers[:4]
        return first + second - spill - capacity
    if scenario == "cause-precedes-effect-v2":
        effect, cause = _clock_times(text)[:2]
        return float(max(0, cause - effect))
    if scenario == "rigid-width-opening-clearance-v2":
        width, opening = numbers[:2]
        return max(0.0, width - opening)
    if scenario == "exclusive-exhaustive-probability-sum-v2":
        probability_a, probability_b, total = numbers[:3]
        return probability_a + probability_b - total
    if scenario == "water-phase-at-temperature-pressure-v2":
        temperature, pressure = numbers[:2]
        if abs(pressure - 1.0) > 0.011:
            return 1.0
        expected = "SOLID ICE" if temperature <= 0.0 else ("LIQUID WATER" if temperature < 100.0 else "WATER VAPOR")
        return 0.0 if f"STATE {expected}" in text else 1.0
    raise ValueError(f"no v2 validator is registered for scenario {scenario!r}")
