import unittest
from cta.transcribed_record_checker import ASSUMPTIONS, check_record
from cta.scei_reasoning_families import FAMILY_IDS, compile_family_record


class TranscribedCheckerTests(unittest.TestCase):
    def test_separate_generated_cases(self):
        # Development fixtures use fresh synthetic keys, never evaluation logs.
        for family in FAMILY_IDS:
            for i in range(12):
                r = compile_family_record('cup', family, variant_key=f'checker-unit-fixture-{i}', seed=314159)
                for state in ('true', 'false'):
                    text = r[state + '_measurement'] + ' | ' + r['uncertainty']
                    prediction = check_record(text, r['assumption'])
                    with self.subTest(family=family, i=i, state=state):
                        self.assertEqual(prediction['prediction'], 'consistent' if state == 'true' else 'inconsistent', prediction)

    def test_duplicate_and_missing_values_abstain(self):
        text = 'CAPACITY 8 L | ADDED 5+4 L | SPILL 1 L | UNCERTAINTY +/- 0.05 L'
        a = ASSUMPTIONS['capacity_conservation']
        self.assertEqual(check_record(text, a)['prediction'], 'consistent')
        self.assertIsNone(check_record(text + ' | SPILL 9 L', a)['prediction'])
        self.assertIsNone(check_record(text.replace('SPILL 1 L', ''), a)['prediction'])

    def test_formula_not_status(self):
        text = 'CAPACITY 8 L | ADDED 5+4 L | SPILL 0 L | UNCERTAINTY +/- 0.05 L'
        a = ASSUMPTIONS['capacity_conservation']
        for suffix in ['', ' | STATUS VALID', ' | VERIFIED CORRECT']:
            self.assertEqual(check_record(text + suffix, a)['prediction'], 'inconsistent')

    def test_clock_and_assumption_boundaries(self):
        a = ASSUMPTIONS['temporal_ledger']
        text = 'START 10:00 | FINISH 11:00 | ELAPSED 60 MIN | CLOCK RESOLUTION +/- 1 MIN'
        self.assertEqual(check_record(text, a)['prediction'], 'consistent')
        self.assertIsNone(check_record(text.replace('11:00', '25:00'), a)['prediction'])
        self.assertIsNone(check_record(text, 'Dates can roll over.')['prediction'])

    def test_phase_boundary_abstains(self):
        text = 'TEMP +000.0 C | PRESSURE 1.00 ATM | STATE SOLID ICE | UNCERTAINTY +/- 0.2 C / +/- 0.01 ATM'
        self.assertIsNone(check_record(text, ASSUMPTIONS['phase_state'])['prediction'])

    def test_invalid_units_and_mixed_schema(self):
        text = 'SOURCE 10 C | CONVERTED 50 MI | ROUNDING TOLERANCE +/- 0.06 MI'
        self.assertIsNone(check_record(text, 'Both fields describe the same quantity; use F=(9/5)C+32.')['prediction'])
        self.assertIsNone(check_record(text + ' | CAPACITY 8 L', '')['prediction'])

    def test_decimal_boundary_and_no_gold_input(self):
        a = ASSUMPTIONS['probability_ledger']
        text = 'P(PASS) 0.33 | P(FAIL) 0.68 | TOTAL 1.00 | ROUNDING +/- 0.01'
        self.assertEqual(check_record(text, a)['prediction'], 'consistent')
        self.assertEqual(check_record(text.replace('0.68', '0.69'), a)['prediction'], 'inconsistent')
        with self.assertRaises(TypeError):
            check_record(text, a, gold='consistent')


if __name__ == '__main__':
    unittest.main()
