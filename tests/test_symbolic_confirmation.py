import copy
import unittest
from cta.scei_reasoning_families import FAMILY_IDS, compile_family_record
from cta.verification_workbench import canonical
from scripts.symbolic_confirmation_lib import ARMS, prompts, parse_reasoned
from scripts.analyze_symbolic_confirmation import calculate


class ConfirmationTests(unittest.TestCase):
    def test_reasoned_parser_requires_unique_final_line(self):
        m = {'A': 'consistent', 'B': 'inconsistent', 'C': 'absent'}
        self.assertEqual(parse_reasoned('8 + 1 = 9.\nFINAL: B', m), 'inconsistent')
        for s in ['A occurs in my reasoning.', 'FINAL: A\nFINAL: B', 'FINAL: A or B', 'Explanation FINAL: B']:
            self.assertIsNone(parse_reasoned(s, m))

    def fixture(self):
        cases, reqs = [], []
        calls = {'qwen7': [], 'qwen3': []}
        m = {'A': 'consistent', 'B': 'inconsistent', 'C': 'absent'}
        for j in range(128):
            i = f'unit-{j:03d}'
            family = FAMILY_IDS[j % 8]
            record = compile_family_record('cup', family, variant_key=i, seed=271828)
            for state, truth, gold in [('record_true', 'true', 'consistent'), ('record_false', 'false', 'inconsistent')]:
                text = record[truth + '_measurement'] + ' | ' + record['uncertainty']
                c = dict(item_id=i, state=state, gold=gold, family=family, source='coco' if j < 64 else 'voc',
                    source_sha256=str(j), image=i+'.png', image_sha256=i, assumption=record['assumption'], fields=text, option_map=m)
                cases.append(c)
                for arm in ARMS:
                    r = dict(key=canonical([i,state,arm]), item_id=i, state=state, arm=arm,
                        image=c['image'], image_sha256=i, prompt=prompts(c['assumption'],m)[arm], tokens=384)
                    reqs.append(r)
                    option = 'A' if truth == 'true' else 'B'
                    raw = text if arm == 'read' else option if arm == 'direct' else 'Calculation\nFINAL: ' + option
                    for model in calls:
                        calls[model].append(dict(key=r['key'], request=r, raw=raw, error=None,
                            started_at='2026-01-02', finished_at='2026-01-02', trace=dict(image_supplied=True,
                                generated_tokens_including_special=10, input_tokens=30, wall_seconds=0.1, hit_output_cap=False)))
        reg = dict(frozen_at='2026-01-01',excluded_item_ids=[],excluded_source_sha256=[])
        return reg,cases,reqs,calls

    def test_complete_replay_and_abstention_denominator(self):
        reg,cases,reqs,calls = self.fixture()
        calls['qwen7'][0]['raw'] = 'unparseable'
        result,predictions = calculate(reg,cases,reqs,calls)
        self.assertEqual(result['actual_calls'],1536)
        self.assertEqual(len(predictions),1536)
        self.assertEqual(result['models']['qwen7']['arms']['direct']['pair_correct'],127)
        self.assertEqual(result['models']['qwen7']['arms']['direct']['n'],256)
        self.assertEqual(result['models']['qwen7']['arms']['read']['pair_correct'],128)
        self.assertEqual(len(result['primary_tests']),4)

    def test_rejects_overlap_and_missing_cells(self):
        reg,cases,reqs,calls = self.fixture()
        reg['excluded_item_ids'] = [cases[0]['item_id']]
        with self.assertRaisesRegex(ValueError,'overlap'):
            calculate(reg,cases,reqs,calls)
        reg['excluded_item_ids'] = []
        calls['qwen3'].pop()
        with self.assertRaisesRegex(ValueError,'coverage'):
            calculate(reg,cases,reqs,calls)

    def test_rejects_budget_and_timing_drift(self):
        reg,cases,reqs,calls = self.fixture()
        changed = copy.deepcopy(reqs)
        changed[0]['tokens'] = 96
        with self.assertRaisesRegex(ValueError,'budget'):
            calculate(reg,cases,changed,calls)
        calls['qwen3'][0]['started_at'] = '2025-01-01'
        with self.assertRaisesRegex(ValueError,'freeze-time'):
            calculate(reg,cases,reqs,calls)


if __name__ == '__main__':
    unittest.main()
