"""Retrospective full-coverage read-plus-symbolic baseline; never new inference."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cta.transcribed_record_checker import check_record, VERSION
from cta.verification_workbench import canonical, parse_option, exact_read
from scripts.analyze_verification_diagnostic import paired_test

ROOT = Path(__file__).resolve().parents[1]
CODE = ['cta/transcribed_record_checker.py', 'scripts/evaluate_transcribed_record_checker.py']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def freeze(source, output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = ['cases.json', 'requests_0.jsonl', 'requests_1.jsonl', 'raw/qwen7.jsonl', 'raw/qwen3.jsonl']
    protocol = {
        'version': VERSION, 'frozen_at': datetime.now(timezone.utc).isoformat(),
        'status': 'retrospective_analysis_specification_not_original_preregistration',
        'source': 'evidence/channel_binding_n128',
        'inputs': {name: sha(source / name) for name in inputs},
        'code': {name: sha(ROOT / name) for name in CODE},
        'selection': 'All 128 media items, both states, both archived Qwen models. No outcome-based selection.',
        'checker_inputs': 'Only the raw independent-read response and the public task assumption. No gold, ID, family label, or target fields.',
        'abstention': 'Missing, duplicate, unsupported, or ambiguous fields abstain; abstentions count incorrect in all-item metrics.',
        'primary_endpoint': 'All-item both-state accuracy; coverage and accuracy on covered cases reported separately.',
        'comparisons': 'Two exploratory paired tests: scene decision versus read-plus-checker, one per model; Holm across two.',
        'uncertainty': 'Paired source-stratified percentile bootstrap, 10000 draws, seed 20260914; exact two-sided McNemar.',
        'cost': 'Reuse 512 archived read responses. Zero new model calls. Reads had cap 384 versus direct decisions cap 96; not compute-matched.',
        'limitations': 'Template-aware and post hoc. Same reused scenes. Neither independent human gold validation nor a new held-out defense confirmation. No object-binding evaluation.',
        'parser_policy': 'Developed from published generator grammar and fresh synthetic unit fixtures before scoring archived responses. No tuning after aggregate evaluation.',
    }
    write(output / 'protocol.json', protocol)
    (output / 'executed_source').mkdir()
    for name in CODE:
        dest = output / 'executed_source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes())
    print('Retrospective protocol frozen. No archived responses scored yet.')


def metrics(rows):
    ids = sorted({r['item_id'] for r in rows})
    covered = [r for r in rows if r['prediction'] is not None]
    correct = sum(r['correct'] for r in rows)
    pair_vector = [int(all(r['correct'] for r in rows if r['item_id'] == i)) for i in ids]
    pair_covered = sum(all(r['prediction'] is not None for r in rows if r['item_id'] == i) for i in ids)
    return {'n': len(rows), 'correct': correct, 'covered': len(covered), 'abstentions': len(rows) - len(covered),
            'selective_accuracy': correct / len(covered) if covered else None,
            'pair_n': len(ids), 'pair_correct': sum(pair_vector), 'pair_covered': pair_covered,
            'pair_vector': pair_vector, 'item_ids': ids,
            'valid_correct': sum(r['correct'] for r in rows if r['gold'] == 'consistent'),
            'false_correct': sum(r['correct'] for r in rows if r['gold'] == 'inconsistent')}


def evaluate(source, output, replay=False):
    protocol = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    for name, h in protocol['inputs'].items():
        if sha(source / name) != h:
            raise ValueError('input drift: ' + name)
    for name, h in protocol['code'].items():
        if sha(ROOT / name) != h or sha(output / 'executed_source' / name) != h:
            raise ValueError('code drift: ' + name)
    cases = [c for c in json.loads((source / 'cases.json').read_text(encoding='utf-8')) if c['study'] == 'media']
    if len(cases) != 256 or len({(c['item_id'], c['state']) for c in cases}) != 256:
        raise ValueError('case coverage')
    requests = {r['key']: r for shard in (0, 1) for r in read_lines(source / f'requests_{shard}.jsonl')}
    result = {'protocol_sha256': sha(output / 'protocol.json'), 'new_model_calls': 0,
              'archived_read_calls': 512, 'models': {}, 'exploratory_tests': []}
    predictions = []
    # Oracle check is implementation sanity only, never substituted into model predictions.
    oracle = [dict(c, prediction=check_record(c['fields'], c['assumption'])['prediction']) for c in cases]
    result['gold_text_implementation_sanity'] = {
        'n': len(oracle), 'agrees_with_nominal_gold': sum(c['prediction'] == c['gold'] for c in oracle),
        'scope': 'Checks code against nominal synthetic labels; not independent human validation.'}
    for model in ('qwen7', 'qwen3'):
        journal = read_lines(source / 'raw' / (model + '.jsonl'))
        calls = {c['key']: c for c in journal}
        if len(calls) != len(journal) or len(journal) != 2304:
            raise ValueError('journal coverage')
        rows, direct = [], []
        for c in cases:
            read = calls[canonical(['media', c['item_id'], c['state'], 'read'])]
            scene = calls[canonical(['media', c['item_id'], c['state'], 'scene'])]
            for call in (read, scene):
                if call['request'] != requests[call['key']]:
                    raise ValueError('request mismatch')
            checked = check_record(read['raw'], c['assumption']) if not read['error'] else {
                'prediction': None, 'reason': 'runtime_error', 'detected_family': None, 'extracted': {}}
            row = {k: c[k] for k in ('item_id', 'state', 'family', 'source', 'gold')}
            row.update(model=model, **checked, correct=int(checked['prediction'] == c['gold']),
                       raw_read=read['raw'], read_call_key=read['key'],
                       exact_read=bool(not read['error'] and exact_read(read['raw'], read['request']['read_target'])))
            rows.append(row)
            p = parse_option(scene['raw'], scene['request']['option_map']) if not scene['error'] else None
            direct.append({**row, 'prediction': p, 'correct': int(p == c['gold'])})
        predictions.extend(rows)
        checked_metrics, direct_metrics = metrics(rows), metrics(direct)
        families = {f: metrics([r for r in rows if r['family'] == f]) for f in sorted({r['family'] for r in rows})}
        result['models'][model] = {'read_plus_checker': checked_metrics, 'direct_scene': direct_metrics,
            'by_family': families, 'abstention_reasons': dict(Counter(r['reason'] for r in rows if r['prediction'] is None)),
            'exact_read_records': sum(r['exact_read'] for r in rows),
            'correct_with_nonexact_read': sum(r['correct'] and not r['exact_read'] for r in rows)}
        strata = [next(r['source'] for r in rows if r['item_id'] == i) for i in checked_metrics['item_ids']]
        test = paired_test(direct_metrics['pair_vector'], checked_metrics['pair_vector'], strata)
        test.update(model=model, a='direct_scene', b='read_plus_checker', status='exploratory_retrospective')
        result['exploratory_tests'].append(test)
    maximum = 0.
    for rank, test in enumerate(sorted(result['exploratory_tests'], key=lambda t: t['p'])):
        maximum = max(maximum, min(1., (2 - rank) * test['p']))
        test['holm_p'] = maximum
    if replay:
        if result != json.loads((output / 'analysis.json').read_text(encoding='utf-8')):
            raise ValueError('analysis replay failed')
        if predictions != read_lines(output / 'predictions.jsonl'):
            raise ValueError('prediction replay failed')
        print('PASS: all 512 archived reads, 256 nominal gold sanity checks, metrics, coverage and exploratory tests replay exactly.')
        return
    if (output / 'analysis.json').exists():
        raise ValueError('analysis exists; use --replay instead of overwriting')
    write(output / 'analysis.json', result)
    (output / 'predictions.jsonl').write_bytes((''.join(canonical(r) + '\n' for r in predictions)).encode('utf-8'))
    lines = [r'\begin{tabular}{llrrrr}', r'\toprule',
             r'Model & Method & Valid & False & Pair & Covered \\', r'\midrule']
    names = {'qwen7': 'Qwen2.5-VL-7B', 'qwen3': 'Qwen3-VL-8B'}
    for model, data in result['models'].items():
        for arm, label in [('direct_scene', 'Direct'), ('read_plus_checker', 'Read + rules')]:
            m = data[arm]
            lines.append(' & '.join([names[model], label, f"{m['valid_correct']}/128", f"{m['false_correct']}/128",
                         f"{m['pair_correct']}/128", f"{m['covered']}/256"]) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (output / 'generated_table.tex').write_bytes(('\n'.join(lines) + '\n').encode('utf-8'))
    write(output / 'provenance.json', {'files': {str(p.relative_to(output)).replace('\\', '/'): sha(p)
         for p in sorted(output.rglob('*')) if p.is_file()}, 'source': protocol['source']})
    print(json.dumps({m: {a: {k: v for k, v in d[a].items() if k not in ('pair_vector', 'item_ids')} for a in ('direct_scene', 'read_plus_checker')}
                      for m, d in result['models'].items()}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=ROOT / 'evidence/channel_binding_n128')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--freeze', action='store_true')
    p.add_argument('--replay', action='store_true')
    a = p.parse_args()
    if a.freeze:
        freeze(a.source, a.output)
    else:
        evaluate(a.source, a.output, a.replay)
