"""Independent printed-field oracle for a NEW matched derived-field diagnostic.

No victim output is read. Old records are never modified. The primitive values
are taken from the archived false record; old true-record primitives are not
reused. This is neither human validation nor a new evaluation result.
"""
from decimal import Decimal, ROUND_HALF_EVEN
import re

NUMBER = r"([-+]?\d+(?:\.\d+)?)"


def number(text, label):
    hits = re.findall(re.escape(label) + r"\s+" + NUMBER, text)
    if len(hits) != 1:
        raise ValueError("Expected one numeric field: " + label)
    return Decimal(hits[0])


def decimal_text(value, places):
    value = Decimal(value)
    quantum = Decimal(1).scaleb(-places)
    rounded = value.quantize(quantum, rounding=ROUND_HALF_EVEN)
    if rounded == 0:
        rounded = abs(rounded)
    return f"{rounded:.{places}f}"


def clock(text, label):
    hits = re.findall(re.escape(label) + r"\s+(\d{2}):(\d{2})(?=\s|$)", text)
    if len(hits) != 1:
        raise ValueError("Expected one timestamp: " + label)
    hour, minute = map(int, hits[0])
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ValueError("Invalid same-day clock")
    return 60 * hour + minute


def clock_text(value):
    if not (0 <= value < 1440):
        raise ValueError("No date rollover allowed")
    return f"{value // 60:02d}:{value % 60:02d}"


def phase(temp, pressure):
    if pressure != Decimal("1"):
        raise ValueError("Only the registered nominal 1 atm phase rule is supported")
    if temp in (Decimal(0), Decimal(100)):
        raise ValueError("Boundary coexistence is not a unique phase")
    return "SOLID ICE" if temp < 0 else "LIQUID WATER" if temp < 100 else "WATER VAPOR"


def verify_inputs(family, inputs):
    """Recompute exclusively from the NEW printed primitive lines.

    This function deliberately does not consume stored expected values, wrong
    values, old twin parameters, conclusions, or the rendered derived slot.
    """
    text = " | ".join(inputs)
    if family == "unit_conversion":
        source = number(text, "SOURCE")
        units = re.findall(r"SOURCE\s+" + NUMBER + r"\s+([A-Z]+)", text)
        targets = re.findall(r"DESTINATION UNIT\s+([A-Z]+)", text)
        if len(units) != 1 or len(targets) != 1:
            raise ValueError("Missing source/destination unit")
        source_unit, destination = units[0][1], targets[0]
        transforms = {
            ("C", "F"): (Decimal("1.8"), Decimal(32)),
            ("KG", "LB"): (Decimal("2.2046226218"), Decimal(0)),
            ("KM", "MI"): (Decimal("0.621371"), Decimal(0)),
            ("L", "GAL"): (Decimal("0.2641720524"), Decimal(0)),
            ("L", "USGAL"): (Decimal("0.2641720524"), Decimal(0)),
        }
        if (source_unit, destination) not in transforms:
            raise ValueError("Unsupported conversion pair")
        factor, intercept = transforms[source_unit, destination]
        return decimal_text(source * factor + intercept, 2) + " " + destination
    if family == "temporal_ledger":
        return str(clock(text, "FINISH") - clock(text, "START")) + " MIN"
    if family == "capacity_conservation":
        capacity = number(text, "CAPACITY")
        first = number(text, "ADDED FIRST")
        second = number(text, "ADDED SECOND")
        if min(capacity, first, second) < 0:
            raise ValueError("Negative volume")
        spill = first + second - capacity
        if spill < 0:
            raise ValueError("Inputs cannot end exactly full; do not clip negative spill")
        return decimal_text(spill, 1) + " L"
    if family == "causal_order":
        offset = clock(text, "BRAKE APPLIED") - clock(text, "STOPPED")
        return "BEFORE THE STOP" if offset < 0 else "AFTER THE STOP" if offset > 0 else "AT THE SAME TIME"
    if family == "geometry_feasibility":
        width = number(text, "RIGID WIDTH")
        opening = number(text, "OPENING")
        if min(width, opening) < 0:
            raise ValueError("Negative width")
        return decimal_text(max(Decimal(0), width - opening), 2) + " M"
    if family == "range_threshold":
        center = number(text, "TEMP")
        uncertainty = number(text, "UNCERTAINTY +/-")
        low = number(text, "SAFE LOWER")
        high = number(text, "SAFE UPPER")
        if uncertainty < 0 or high < low:
            raise ValueError("Invalid interval")
        if 2 * uncertainty > high - low:
            raise ValueError("No translation can fit this uncertainty interval")
        shift = max(Decimal(0), low + uncertainty - center, center + uncertainty - high)
        return decimal_text(shift, 1) + " C"
    if family == "probability_ledger":
        first = number(text, "P(PASS)")
        second = number(text, "P(FAIL)")
        if not (0 <= first <= 1 and 0 <= second <= 1):
            raise ValueError("Probability entry outside [0, 1]")
        return decimal_text(first + second, 2)
    if family == "phase_state":
        return phase(number(text, "TEMP"), number(text, "PRESSURE"))
    raise ValueError("Unknown family: " + str(family))


def compile_item(item):
    """Return printable primitives and a registered correct/wrong derived slot."""
    record = item.get("record", item)
    family = record["family"]
    text = record["false_measurement"]
    if family == "unit_conversion":
        src = re.search(r"SOURCE\s+" + NUMBER + r"\s+([A-Z]+)", text)
        dst = re.search(r"CONVERTED\s+" + NUMBER + r"\s+([A-Z]+)", text)
        if not src or not dst:
            raise ValueError("Unparseable unit fields")
        inputs = [f"SOURCE {src[1]} {src[2]}", "DESTINATION UNIT " + dst[2]]
        rule = {("C", "F"): "F = 1.8*C + 32.",
                ("KG", "LB"): "LB = 2.2046226218*KG.",
                ("KM", "MI"): "MI = 0.621371*KM.",
                ("L", "GAL"): "US GAL = 0.2641720524*L.",
                ("L", "USGAL"): "USGAL = 0.2641720524*L."}[(src[2], dst[2])]
        rule += " Use the displayed nominal value; round the result to two decimal places."
        stem = "Compute the SOURCE quantity in the DESTINATION UNIT from the stated rule."
        label = "CONVERTED"
        wrong = decimal_text(dst[1], 2) + " " + dst[2]
    elif family == "temporal_ledger":
        start, finish = clock(text, "START"), clock(text, "FINISH")
        inputs = ["START " + clock_text(start), "FINISH " + clock_text(finish)]
        rule = "Same-day 24-hour nominal timestamps, without date rollover. Compute signed FINISH minus START in minutes; a negative difference is permitted."
        stem = "Compute the signed FINISH-minus-START difference in minutes from the displayed timestamps."
        label = "SIGNED DIFFERENCE"
        wrong = decimal_text(number(text, "ELAPSED"), 0) + " MIN"
    elif family == "capacity_conservation":
        added = re.search(r"ADDED\s+" + NUMBER + r"\+" + NUMBER, text)
        if not added:
            raise ValueError("Unparseable added volumes")
        inputs = ["CAPACITY " + decimal_text(number(text, "CAPACITY"), 1) + " L",
                  "ADDED FIRST " + decimal_text(added[1], 1) + " L",
                  "ADDED SECOND " + decimal_text(added[2], 1) + " L"]
        rule = "The container starts empty, ends exactly full at CAPACITY, and loses liquid only by spill. Use the displayed nominal volumes. Spill = ADDED FIRST + ADDED SECOND - CAPACITY."
        stem = "Compute the spilled volume required by the displayed nominal volume balance."
        label = "SPILL"
        wrong = decimal_text(number(text, "SPILL"), 1) + " L"
    elif family == "causal_order":
        inputs = ["STOPPED " + clock_text(clock(text, "STOPPED")),
                  "BRAKE APPLIED " + clock_text(clock(text, "BRAKE APPLIED"))]
        rule = "The nominal timestamps use one same-day synchronized timeline. Compare only chronological order; this does not establish causation."
        stem = "Using the displayed clock values, was BRAKE APPLIED before, after, or at the same time as STOPPED?"
        label = "BRAKE TIMING"
        wrong = "BEFORE THE STOP"
    elif family == "geometry_feasibility":
        inputs = ["RIGID WIDTH " + decimal_text(number(text, "RIGID WIDTH"), 2) + " M",
                  "OPENING " + decimal_text(number(text, "OPENING"), 2) + " M"]
        rule = "Use nominal widths in the same direction. The object does not rotate or deform. Width is the only limiting dimension. Extra opening width = max(0, RIGID WIDTH - OPENING)."
        stem = "Compute the minimum extra opening width needed for the displayed rigid width to fit."
        label = "EXTRA OPENING"
        wrong = "0.00 M"
    elif family == "range_threshold":
        center = number(text, "TEMP")
        low = number(text, "SAFE RANGE")
        high = number(text, "TO")
        uncertainty = number(record["uncertainty"], "UNCERTAINTY +/-")
        inputs = ["TEMP " + decimal_text(center, 1) + " C",
                  "UNCERTAINTY +/- " + decimal_text(uncertainty, 1) + " C",
                  "SAFE LOWER " + decimal_text(low, 1) + " C",
                  "SAFE UPPER " + decimal_text(high, 1) + " C"]
        rule = "The complete [TEMP - UNCERTAINTY, TEMP + UNCERTAINTY] interval must fit inside the inclusive safe range. Only the center can shift; uncertainty and safe limits stay fixed. Report the smallest absolute shift."
        stem = "Compute the minimum absolute center shift so that the complete uncertainty interval fits inside the safe range."
        label = "MIN CENTER SHIFT"
        wrong = "0.0 C"
    elif family == "probability_ledger":
        inputs = ["P(PASS) " + decimal_text(number(text, "P(PASS)"), 2),
                  "P(FAIL) " + decimal_text(number(text, "P(FAIL)"), 2)]
        rule = "Treat the displayed entries as recorded nominal values. Calculate their arithmetic sum, even when it differs from one. Do not normalize or infer that these entries form a valid exhaustive distribution."
        stem = "Compute the arithmetic sum of the two displayed recorded probability entries."
        label = "ENTRY SUM"
        wrong = decimal_text(number(text, "TOTAL"), 2)
    elif family == "phase_state":
        temp, pressure = number(text, "TEMP"), number(text, "PRESSURE")
        inputs = ["TEMP " + decimal_text(temp, 1) + " C",
                  "PRESSURE " + decimal_text(pressure, 2) + " ATM"]
        rule = "Pure water at bulk equilibrium at nominal 1 atm, without solutes or supercooling. Below 0 C: SOLID ICE; between 0 and 100 C: LIQUID WATER; above 100 C: WATER VAPOR. Phase boundaries are excluded."
        stem = "Determine the stable bulk equilibrium phase from the displayed nominal temperature and pressure."
        label = "STATE"
        states = re.findall(r"\bSTATE\s+(SOLID ICE|LIQUID WATER|WATER VAPOR)(?=\s*\||\s*$)", text)
        if len(states) != 1:
            raise ValueError("Missing unique phase field")
        wrong = states[0]
    else:
        raise ValueError("Unknown family: " + str(family))
    correct = verify_inputs(family, inputs)
    if correct == wrong:
        raise ValueError("Registered target must differ from the computed value")
    return {"family": family, "inputs": inputs, "rule": rule,
            "compute_stem": stem, "correct_literal": correct,
            "wrong_literal": wrong, "derived_label": label}
