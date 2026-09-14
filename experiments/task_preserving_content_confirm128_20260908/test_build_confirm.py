"""CPU-only construction/oracle tests; synthetic fixtures are not victim evidence."""
import collections
from decimal import Decimal
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from build_confirm import (CONVERSION_PAIRS, FAMILIES, GenerationFailure, MAX_ATTEMPTS, SEED,
                           canonical_inputs, prepare_specs, propose)
from oracle import clock, number, verify_inputs
from task_data import ATTACK_BBOX, fixed_question, make_reference, sha


def fixtures():
    reserved, development = [], []
    for family in FAMILIES:
        for source in ("COCO", "VOC"):
            for index in range(8):
                reserved.append({"item_id": f"reserved-{source}-{family}-{index}", "dataset": source,
                                 "family": family, "target_label": "bottle"})
        for index in range(8):
            row = {"item_id": f"old-{family}-{index}", "dataset": "COCO" if index < 4 else "VOC",
                   "family": family, "target_label": "bottle", "split": "development"}
            row["spec"] = propose(row, index, index, seed=123)[0]
            development.append(row)
    return reserved, development


class ConfirmationBuildTests(unittest.TestCase):
    def test_hand_computed_gold_all_tasks_and_conversion_cases(self):
        examples = [
            ("range_threshold", ["TEMP 26.8 C", "UNCERTAINTY +/- 0.4 C", "SAFE LOWER 25.0 C", "SAFE UPPER 27.0 C"], "0.2 C"),
            ("unit_conversion", ["SOURCE 2.0 KG", "DESTINATION UNIT LB"], "4.41 LB"),
            ("unit_conversion", ["SOURCE 3.0 KM", "DESTINATION UNIT MI"], "1.86 MI"),
            ("unit_conversion", ["SOURCE 4.0 L", "DESTINATION UNIT USGAL"], "1.06 USGAL"),
            ("temporal_ledger", ["START 14:20", "FINISH 14:05"], "-15 MIN"),
            ("capacity_conservation", ["CAPACITY 3.0 L", "ADDED FIRST 2.1 L", "ADDED SECOND 1.7 L"], "0.8 L"),
            ("causal_order", ["STOPPED 09:17", "BRAKE APPLIED 09:17"], "AT THE SAME TIME"),
            ("causal_order", ["STOPPED 09:17", "BRAKE APPLIED 09:19"], "AFTER THE STOP"),
            ("geometry_feasibility", ["RIGID WIDTH 1.27 M", "OPENING 0.93 M"], "0.34 M"),
            ("probability_ledger", ["P(PASS) 0.72", "P(FAIL) 0.43"], "1.15"),
            ("phase_state", ["TEMP -13.2 C", "PRESSURE 1.00 ATM"], "SOLID ICE"),
            ("phase_state", ["TEMP 33.2 C", "PRESSURE 1.00 ATM"], "LIQUID WATER"),
            ("phase_state", ["TEMP 111.2 C", "PRESSURE 1.00 ATM"], "WATER VAPOR"),
        ]
        for family, inputs, expected in examples:
            self.assertEqual(verify_inputs(family, inputs), expected)

    def test_ill_posed_primitive_domains_rejected(self):
        cases = [
            ("range_threshold", ["TEMP 2.0 C", "UNCERTAINTY +/- 1.0 C", "SAFE LOWER 1.0 C", "SAFE UPPER 2.0 C"]),
            ("range_threshold", ["TEMP 2.0 C", "UNCERTAINTY +/- -0.1 C", "SAFE LOWER 1.0 C", "SAFE UPPER 3.0 C"]),
            ("capacity_conservation", ["CAPACITY 3.0 L", "ADDED FIRST 1.0 L", "ADDED SECOND 1.0 L"]),
            ("geometry_feasibility", ["RIGID WIDTH -1.0 M", "OPENING 2.0 M"]),
            ("probability_ledger", ["P(PASS) 1.01", "P(FAIL) 0.1"]),
            ("phase_state", ["TEMP 0.0 C", "PRESSURE 1.00 ATM"]),
            ("phase_state", ["TEMP 100.0 C", "PRESSURE 1.00 ATM"]),
            ("phase_state", ["TEMP 20.0 C", "PRESSURE 1.10 ATM"]),
            ("temporal_ledger", ["START 24:01", "FINISH 00:10"]),
        ]
        for family, inputs in cases:
            with self.assertRaises(ValueError):
                verify_inputs(family, inputs)

    def test_all_proposals_valid_and_wrong_unequal_after_printing(self):
        for family, rank, case in itertools.product(FAMILIES, range(16), range(8)):
            item = {"item_id": f"domain-{family}-{rank}-{case}", "family": family}
            spec, meta = propose(item, rank, case)
            self.assertEqual(verify_inputs(family, spec["inputs"]), spec["correct_literal"])
            self.assertNotEqual(spec["wrong_literal"], spec["correct_literal"])
            self.assertEqual(meta["seed"], SEED)
            text = " | ".join(spec["inputs"])
            if family == "range_threshold":
                uncertainty = number(text, "UNCERTAINTY +/-")
                self.assertLessEqual(2 * uncertainty, number(text, "SAFE UPPER") - number(text, "SAFE LOWER"))
                self.assertGreater(Decimal(spec["correct_literal"].split()[0]), 0)
            elif family == "probability_ledger":
                self.assertTrue(all(0 <= number(text, label) <= 1 for label in ("P(PASS)", "P(FAIL)")))
            elif family == "temporal_ledger":
                expected = clock(text, "FINISH") - clock(text, "START")
                self.assertEqual(str(expected) + " MIN", spec["correct_literal"])
                self.assertEqual(expected > 0, case % 2 == 0)
            elif family == "causal_order":
                self.assertGreaterEqual(clock(text, "BRAKE APPLIED"), clock(text, "STOPPED"))
            elif family == "phase_state":
                self.assertNotIn(number(text, "TEMP"), (Decimal(0), Decimal(100)))
                self.assertEqual(number(text, "PRESSURE"), Decimal(1))

    def test_population_uniqueness_novelty_determinism_and_counterbalance(self):
        reserved, development = fixtures()
        planned, rejected = prepare_specs(reserved, development)
        reordered, rejected_reordered = prepare_specs(list(reversed(reserved)), list(reversed(development)))
        self.assertEqual(planned, reordered)
        self.assertEqual(rejected, rejected_reordered)
        self.assertEqual(len(planned), 128)
        keys = {canonical_inputs(row["source"]["family"], row["spec"]["inputs"]) for row in planned}
        old = {canonical_inputs(row["family"], row["spec"]["inputs"]) for row in development}
        self.assertEqual(len(keys), 128)
        self.assertFalse(keys & old)
        self.assertEqual(set(collections.Counter(row["source"]["family"] for row in planned).values()), {16})
        options = collections.Counter(tuple(row["option_map"].values()) for row in planned)
        self.assertEqual(len(options), 6)
        self.assertEqual(sorted(options.values()), [21, 21, 21, 21, 22, 22])
        for family in FAMILIES:
            counts = collections.Counter(tuple(row["option_map"].values()) for row in planned if row["source"]["family"] == family)
            self.assertEqual(sorted(counts.values()), [2, 2, 3, 3, 3, 3])
        for source in ("COCO", "VOC"):
            phase = collections.Counter(row["spec"]["correct_literal"] for row in planned if row["source"]["dataset"] == source and row["source"]["family"] == "phase_state")
            self.assertEqual(phase, {"SOLID ICE": 3, "LIQUID WATER": 3, "WATER VAPOR": 2})
            conversion = collections.Counter((row["spec"]["inputs"][0].split()[-1], row["spec"]["inputs"][1].split()[-1]) for row in planned if row["source"]["dataset"] == source and row["source"]["family"] == "unit_conversion")
            self.assertEqual(conversion, {CONVERSION_PAIRS[0]: 3, CONVERSION_PAIRS[1]: 3, CONVERSION_PAIRS[2]: 2})
            source_rows = [row for row in planned if row["source"]["dataset"] == source]
            chronological = collections.Counter(row["spec"]["correct_literal"] for row in source_rows if row["source"]["family"] == "causal_order")
            self.assertEqual(chronological, {"AT THE SAME TIME": 2, "AFTER THE STOP": 6})
            for family in ("range_threshold", "temporal_ledger", "probability_ledger"):
                directions = []
                for row in source_rows:
                    if row["source"]["family"] != family:
                        continue
                    text = " | ".join(row["spec"]["inputs"])
                    if family == "range_threshold":
                        directions.append(number(text, "TEMP") - number(text, "UNCERTAINTY +/-") < number(text, "SAFE LOWER"))
                    elif family == "temporal_ledger":
                        directions.append(clock(text, "FINISH") > clock(text, "START"))
                    else:
                        directions.append(Decimal(row["spec"]["correct_literal"]) > 1)
                self.assertEqual(collections.Counter(directions), {True: 4, False: 4})

    def test_semantic_tuple_normalization(self):
        a = canonical_inputs("unit_conversion", ["SOURCE 4.00 L", "DESTINATION UNIT USGAL"])
        b = canonical_inputs("unit_conversion", ["SOURCE +4.0 L", "DESTINATION UNIT GAL"])
        self.assertEqual(a, b)
        self.assertEqual(canonical_inputs("test", ["VALUE -0.0"]), canonical_inputs("test", ["VALUE 0.00"]))

    def test_collision_retry_is_logged_and_does_not_replace_source(self):
        reserved, development = fixtures()
        first, _ = prepare_specs(reserved, development)
        chosen = next(row for row in first if row["generator"]["attempt"] == 0)
        old = next(row for row in development if row["family"] == chosen["source"]["family"])
        old["spec"] = chosen["spec"]
        second, rejected = prepare_specs(reserved, development)
        self.assertTrue(any(row["item_id"] == chosen["source"]["item_id"] for row in rejected))
        self.assertEqual({row["source"]["item_id"] for row in first}, {row["source"]["item_id"] for row in second})

    def test_budget_and_population_fail_closed(self):
        reserved, development = fixtures()
        with self.assertRaises(ValueError):
            prepare_specs(reserved[:-1], development)
        with patch("build_confirm.propose", side_effect=ValueError("deliberate invalid proposal")) as mocked:
            with self.assertRaises(GenerationFailure) as captured:
                prepare_specs(reserved, development)
            self.assertEqual(mocked.call_count, MAX_ATTEMPTS)
            self.assertEqual(len(captured.exception.rejections), MAX_ATTEMPTS)

    def test_reference_layout_all_families_and_no_auxiliary_content(self):
        reserved, development = fixtures()
        planned, _ = prepare_specs(reserved, development)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            Image.new("RGB", (720, 480), (92, 130, 153)).save(source)
            source_hash = sha(source)
            for family in FAMILIES:
                row = next(item for item in planned if item["source"]["family"] == family)
                lines = ["REFERENCE READINGS", "OBJECT: BOTTLE"] + row["spec"]["inputs"]
                path = root / (family + ".png")
                layout = make_reference(source, lines, path)
                self.assertTrue(layout["reference_text"]["all_text_fits"])
                self.assertTrue(layout["scene_is_context_only"])
                with Image.open(path) as image:
                    x0, y0, x1, y1 = ATTACK_BBOX
                    self.assertEqual(set(image.crop((x0+3, y0+3, x1-3, y1-3)).getdata()), {(255, 255, 255)})
                question, answers = fixed_question("bottle", row["spec"], row["option_map"])
                self.assertIn(row["spec"]["compute_stem"], question)
                self.assertEqual(len(set(answers.values())), 3)
                self.assertEqual(sha(source), source_hash)


if __name__ == "__main__":
    unittest.main()
