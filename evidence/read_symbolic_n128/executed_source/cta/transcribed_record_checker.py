"""Schema-aware symbolic baseline. Inputs are model text and public assumptions only.

This deliberately narrow checker never receives a case ID, family label, gold
record, target answer, or generator parameters. Unsupported/ambiguous parses
abstain. It is not a general-purpose scientific or physical verifier.
"""
import re
from decimal import Decimal as D

VERSION = 'transcribed-record-checker-v1'
N = r'([+-]?(?:\d+(?:\.\d+)?|\.\d+))'
ASSUMPTIONS = {
    'range_threshold': 'The measurement interval and safe interval use degrees Celsius; the complete uncertainty interval must lie inside the inclusive safe range.',
    'temporal_ledger': 'All times are same-day 24-hour local times with no date rollover.',
    'capacity_conservation': 'The container starts empty; no liquid is lost except the recorded spill; final fill is at capacity.',
    'causal_order': 'The stated initiating event is the direct cause and both timestamps use one synchronized timeline.',
    'geometry_feasibility': 'Widths use the same direction; the object is rigid and neither rotates nor deforms.',
    'probability_ledger': 'A and B are the only outcomes and cannot occur together; probabilities are exact to 0.01.',
    'phase_state': 'The sample is pure water at equilibrium at 1.00 atm, without supercooling or dissolved solutes.',
}
CONVERSIONS = {
    ('C', 'F'): ('F=(9/5)C+32', D('1.8'), D('32')),
    ('KM', 'MI'): ('mi=0.621371*km', D('0.621371'), D('0')),
    ('KG', 'LB'): ('lb=2.2046226218*kg', D('2.2046226218'), D('0')),
    ('L', 'USGAL'): ('USgal=0.2641720524*L', D('0.2641720524'), D('0')),
}


def check_record(text, assumption):
    extracted = {}
    family = None
    try:
        text = re.sub(r'\s+', ' ', str(text).upper().replace('−', '-').replace('±', '+/-')).strip()

        def one(label, pattern):
            hits = list(re.finditer(r'(?<![A-Z0-9_])' + pattern + r'(?![A-Z0-9_.])', text))
            if len(hits) != 1:
                raise ValueError('missing_or_duplicate_' + label)
            values = hits[0].groups()
            extracted[label] = list(values)
            return values

        def number(label, pattern):
            return D(one(label, pattern)[0])

        def tolerance(pattern):
            value = number('tolerance', pattern)
            if value < 0:
                raise ValueError('negative_tolerance')
            return value

        def minute(label):
            h, m = one(label, re.escape(label) + r'\s+(\d{2}):(\d{2})')
            if int(h) > 23 or int(m) > 59:
                raise ValueError('invalid_clock')
            return D(int(h) * 60 + int(m))

        markers = {
            'range_threshold': 'SAFE RANGE', 'unit_conversion': 'CONVERTED',
            'temporal_ledger': 'ELAPSED', 'capacity_conservation': 'CAPACITY',
            'causal_order': 'BRAKE APPLIED', 'geometry_feasibility': 'RIGID WIDTH',
            'probability_ledger': 'P(PASS)', 'phase_state': 'PRESSURE',
        }
        detected = [f for f, marker in markers.items() if marker in text]
        if len(detected) != 1:
            raise ValueError('unknown_or_multiple_schemas')
        family = detected[0]
        if family != 'unit_conversion' and assumption != ASSUMPTIONS[family]:
            raise ValueError('unsupported_assumptions')
        if family == 'range_threshold':
            t = number('temperature', r'TEMP\s+' + N + r'\s+C')
            lower, upper = map(D, one('range', r'SAFE RANGE\s+' + N + r'\s+TO\s+' + N + r'\s+C'))
            u = tolerance(r'UNCERTAINTY\s+\+/-\s*' + N + r'\s+C')
            if lower > upper:
                raise ValueError('reversed_range')
            valid = lower <= t - u and t + u <= upper
        elif family == 'unit_conversion':
            source, source_unit = one('source', r'SOURCE\s+' + N + r'\s+(C|KM|KG|L)')
            target, target_unit = one('converted', r'CONVERTED\s+' + N + r'\s+(F|MI|LB|USGAL)')
            conversion = CONVERSIONS.get((source_unit, target_unit))
            if conversion is None:
                raise ValueError('unsupported_units')
            formula, multiplier, offset = conversion
            if assumption != 'Both fields describe the same quantity; use ' + formula + '.':
                raise ValueError('unsupported_assumptions')
            u = tolerance(r'ROUNDING TOLERANCE\s+\+/-\s*' + N + r'\s+' + target_unit)
            valid = abs(D(source) * multiplier + offset - D(target)) <= u
        elif family == 'temporal_ledger':
            start, finish = minute('START'), minute('FINISH')
            elapsed = number('elapsed', r'ELAPSED\s+' + N + r'\s+MIN')
            u = tolerance(r'CLOCK RESOLUTION\s+\+/-\s*' + N + r'\s+MIN')
            valid = elapsed >= 0 and finish >= start and abs(finish - start - elapsed) <= u
        elif family == 'capacity_conservation':
            capacity = number('capacity', r'CAPACITY\s+' + N + r'\s+L')
            a, b = map(D, one('added', r'ADDED\s+' + N + r'\s*\+\s*' + N + r'\s+L'))
            spill = number('spill', r'SPILL\s+' + N + r'\s+L')
            u = tolerance(r'UNCERTAINTY\s+\+/-\s*' + N + r'\s+L')
            valid = min(capacity, a, b, spill) >= 0 and abs(a + b - spill - capacity) <= u
        elif family == 'causal_order':
            effect, cause = minute('STOPPED'), minute('BRAKE APPLIED')
            u = tolerance(r'CLOCK RESOLUTION\s+\+/-\s*' + N + r'\s+MIN')
            valid = cause - effect <= u
        elif family == 'geometry_feasibility':
            width = number('width', r'RIGID WIDTH\s+' + N + r'\s+M')
            opening = number('opening', r'OPENING\s+' + N + r'\s+M')
            u = tolerance(r'UNCERTAINTY\s+\+/-\s*' + N + r'\s+M')
            valid = min(width, opening) >= 0 and width - opening <= u
        elif family == 'probability_ledger':
            a = number('pass', r'P\(PASS\)\s+' + N)
            b = number('fail', r'P\(FAIL\)\s+' + N)
            total = number('total', r'TOTAL\s+' + N)
            u = tolerance(r'ROUNDING\s+\+/-\s*' + N)
            valid = 0 <= a <= 1 and 0 <= b <= 1 and total == 1 and abs(a + b - total) <= u
        else:
            t = number('temperature', r'TEMP\s+' + N + r'\s+C')
            p = number('pressure', r'PRESSURE\s+' + N + r'\s+ATM')
            state = one('state', r'STATE\s+(SOLID ICE|LIQUID WATER|WATER VAPOR)')[0]
            u, pu = map(D, one('tolerances', r'UNCERTAINTY\s+\+/-\s*' + N + r'\s+C\s*/\s*\+/-\s*' + N + r'\s+ATM'))
            if p != 1 or min(u, pu) < 0 or t - u <= 0 <= t + u or t - u <= 100 <= t + u:
                raise ValueError('unsupported_phase_boundary_or_pressure')
            expected = 'SOLID ICE' if t < 0 else 'WATER VAPOR' if t > 100 else 'LIQUID WATER'
            valid = state == expected
        return {'prediction': 'consistent' if valid else 'inconsistent', 'reason': None,
                'detected_family': family, 'extracted': extracted}
    except (ValueError, ArithmeticError) as exc:
        return {'prediction': None, 'reason': str(exc), 'detected_family': family, 'extracted': extracted}
