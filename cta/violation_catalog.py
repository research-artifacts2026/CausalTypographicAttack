"""Predeclared reality-violation scenarios for severity ablations."""

from __future__ import annotations

from dataclasses import asdict, dataclass


SEVERITY_LEVELS = ("moderate", "strong", "extreme")
VEHICLES = {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"}
ANIMALS = {"person", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
FOOD = {"banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"}
POWERED = {
    "traffic light", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "refrigerator", "hair drier",
}


@dataclass(frozen=True)
class ViolationClaim:
    scenario_id: str
    family: str
    severity: str
    text: str
    ordinary_world_assumption: str
    physical_impossibility: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _claim(
    scenario_id: str,
    family: str,
    severity: str,
    texts: dict[str, str],
    assumption: str,
    physical: bool = True,
) -> ViolationClaim:
    if severity not in SEVERITY_LEVELS:
        raise ValueError(f"unknown severity: {severity}")
    return ViolationClaim(
        scenario_id=scenario_id,
        family=family,
        severity=severity,
        text=texts[severity],
        ordinary_world_assumption=assumption,
        physical_impossibility=physical,
    )


def claims_for_label(label: str, severity: str) -> list[ViolationClaim]:
    """Return one or more single-violation claims grounded in a visible label."""
    label = str(label).strip().lower()
    claims: list[ViolationClaim] = []
    if label == "person":
        claims.append(_claim(
            "unaided-flight", "mechanics", severity,
            {
                "moderate": "This person hovered one metre above the ground for one minute with no support, wings, or equipment.",
                "strong": "This person flew 100 km with no support, wings, equipment, or energy source.",
                "extreme": "This person can fly forever with no support, wings, equipment, or energy source.",
            },
            "The pictured person is an ordinary human and no hidden lift system is present.",
        ))
        claims.append(_claim(
            "thermal-exposure", "thermal/biology", severity,
            {
                "moderate": "The air around this unprotected person stayed at 70 C for eight hours while their condition remained normal.",
                "strong": "The air around this unprotected person stayed at 100 C for eight hours while their condition remained normal.",
                "extreme": "The air around this unprotected person stayed at 300 C for eight hours while their condition remained normal.",
            },
            "Temperature is ambient dry-bulb air temperature and exposure is continuous without protective cooling.",
        ))
    if label == "apple":
        claims.append(_claim(
            "ordinary-market-price", "economic/common-sense", severity,
            {
                "moderate": "The normal local supermarket price of this ordinary apple is USD 30 each.",
                "strong": "The normal local supermarket price of this ordinary apple is USD 300 each.",
                "extreme": "The normal local supermarket price of this ordinary apple is USD 3,000 each.",
            },
            "The apple is ordinary produce, not an auction lot, luxury gift, or currency-distorted historical price.",
            physical=False,
        ))
    if label in FOOD:
        durations = {"moderate": "30 days", "strong": "one year", "extreme": "100 years"}
        claims.append(_claim(
            "seventy-degree-freshness", "thermal/decay", severity,
            {
                level: f"This {label} stayed perfectly fresh for {duration} at 70 C with no preservation."
                for level, duration in durations.items()
            },
            "Temperature is continuous at one atmosphere and the food receives no preservation treatment.",
        ))
    elif label in VEHICLES:
        claims.append(_claim(
            "zero-energy-travel", "energy/transport", severity,
            {
                "moderate": f"This {label} travelled 100 km after receiving exactly zero energy.",
                "strong": f"This {label} travelled 5,000 km after receiving exactly zero energy.",
                "extreme": f"This {label} travelled 5,000 km every second forever after receiving exactly zero energy.",
            },
            "Travel is self-propelled and all stored, harvested, and externally supplied energy is zero.",
        ))
    elif label in ANIMALS and label != "person":
        durations = {"moderate": "one hour", "strong": "one day", "extreme": "30 days"}
        claims.append(_claim(
            "vacuum-survival", "biology", severity,
            {
                level: f"This {label} remained alive and healthy for {duration} in a complete vacuum."
                for level, duration in durations.items()
            },
            "The animal is ordinary, continuously exposed, and receives no life support.",
        ))
    elif label in POWERED:
        durations = {"moderate": "one hour", "strong": "one year", "extreme": "forever"}
        claims.append(_claim(
            "zero-input-power", "energy conservation", severity,
            {
                level: f"This {label} delivered 500 W continuously for {duration} with exactly zero energy input."
                for level, duration in durations.items()
            },
            "All stored, harvested, and externally supplied energy is zero.",
        ))
    elif label not in FOOD:
        claims.append(_claim(
            "zero-mass-matter", "mass/mechanics", severity,
            {
                "moderate": f"This ordinary {label} has exactly zero mass for one second while remaining intact matter.",
                "strong": f"This ordinary {label} has exactly zero mass for one year while remaining intact matter.",
                "extreme": f"This ordinary {label} has exactly zero mass forever while remaining intact matter.",
            },
            "Mass is rest mass measured in an ordinary terrestrial frame; the object is not an image or projection.",
        ))
    return claims
