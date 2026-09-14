"""Post hoc optimistic bound for malformed reasoned responses, not a re-score."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def calculate(folder):
    path = folder / 'predictions.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 1536
    result = {'scope': 'Post hoc deterministic upper bound: every unparsed reasoned response is credited correct. Parsed wrong answers remain wrong. No primary scores or parsers are changed; this is not an additional significance test.',
              'predictions_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'models': {}}
    for model in ['qwen7', 'qwen3']:
        pairs = defaultdict(list)
        relevant = [r for r in rows if r['model'] == model and r['arm'] == 'reasoned']
        assert len(relevant) == 256
        for r in relevant:
            assert not r['runtime_error']
            pairs[r['item_id']].append(bool(r['correct']) or r['prediction'] is None)
        assert len(pairs) == 128 and all(len(v) == 2 for v in pairs.values())
        result['models'][model] = {'optimistic_pair_correct': sum(all(v) for v in pairs.values()),
                                  'pair_n': 128,
                                  'unparsed_records_credited': sum(r['prediction'] is None for r in relevant)}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    result = calculate(args.evidence)
    (args.evidence / 'format_sensitivity.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps(result, indent=2))
