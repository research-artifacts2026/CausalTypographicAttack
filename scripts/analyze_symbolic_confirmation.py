"""Audit and replay all registered symbolic-confirmation calls and tests."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cta.verification_workbench import canonical, digest, sha, read_jsonl
from cta.transcribed_record_checker import check_record
from scripts.symbolic_confirmation_lib import ARMS, parse_option, parse_reasoned, prompts
from scripts.analyze_verification_diagnostic import paired_test


def write(path, obj):
    path.write_bytes((json.dumps(obj, indent=2, ensure_ascii=False) + '\n').encode())


def calculate(reg, cases, requests, model_calls):
    ci = {(c['item_id'], c['state']): c for c in cases}
    expected = {r['key']: r for r in requests}
    ids = sorted({c['item_id'] for c in cases})
    if len(cases) != 256 or len(ci) != 256 or len(ids) != 128 or len(requests) != 768 or len(expected) != 768:
        raise ValueError('case/request coverage')
    if set(ids) & set(reg['excluded_item_ids']) or {c['source_sha256'] for c in cases} & set(reg['excluded_source_sha256']):
        raise ValueError('source overlap')
    if len({c['source_sha256'] for c in cases}) != 128:
        raise ValueError('duplicate source')
    for i in ids:
        a, b = ci[i, 'record_true'], ci[i, 'record_false']
        if a['source_sha256'] != b['source_sha256'] or a['assumption'] != b['assumption'] or a['option_map'] != b['option_map']:
            raise ValueError('twin identity mismatch')
    for r in requests:
        c = ci[r['item_id'], r['state']]
        if r['prompt'] != prompts(c['assumption'], c['option_map'])[r['arm']] or r['tokens'] != 384:
            raise ValueError('prompt or budget drift')
        if r['image_sha256'] != c['image_sha256'] or r['image'] != c['image']:
            raise ValueError('image association drift')
    result = {'status': 'complete', 'actual_calls': 0, 'models': {}, 'primary_tests': []}
    predictions = []
    for model in ['qwen7', 'qwen3']:
        calls = model_calls[model]
        if len(calls) != 768 or len({c['key'] for c in calls}) != 768 or {c['key'] for c in calls} != set(expected):
            raise ValueError('call coverage')
        rows = []
        for call in calls:
            r = expected[call['key']]
            if call['request'] != r or call['started_at'] < reg['frozen_at']:
                raise ValueError('call request or freeze-time mismatch')
            c = ci[r['item_id'], r['state']]
            if not call['error']:
                trace = call['trace']
                if trace.get('image_supplied') is not True or not 1 <= trace['generated_tokens_including_special'] <= 384:
                    raise ValueError('missing or invalid inference trace')
            checked = None
            if call['error']:
                prediction = None
            elif r['arm'] == 'read':
                checked = check_record(call['raw'], c['assumption'])
                prediction = checked['prediction']
            elif r['arm'] == 'reasoned':
                prediction = parse_reasoned(call['raw'], c['option_map'])
            else:
                prediction = parse_option(call['raw'], c['option_map'])
            rows.append({'model': model, 'item_id': c['item_id'], 'state': c['state'], 'arm': r['arm'],
                'gold': c['gold'], 'prediction': prediction, 'correct': int(prediction == c['gold']),
                'checker': checked, 'runtime_error': bool(call['error']), 'trace': call['trace']})
        predictions.extend(rows)
        mr = {}
        strata = [ci[i, 'record_true']['source'] + '/' + ci[i, 'record_true']['family'] for i in ids]
        for arm in ARMS:
            group = [r for r in rows if r['arm'] == arm]
            ix = {(r['item_id'], r['state']): r for r in group}
            vector = [int(all(ix[i, s]['correct'] for s in ['record_true', 'record_false'])) for i in ids]
            good_traces = [r['trace'] for r in group if not r['runtime_error']]
            def avg(key):
                return sum(t[key] for t in good_traces) / len(good_traces) if good_traces else None
            mr[arm] = {'correct': sum(r['correct'] for r in group), 'n': len(group),
                'valid_correct': sum(r['correct'] for r in group if r['state'] == 'record_true'),
                'false_correct': sum(r['correct'] for r in group if r['state'] == 'record_false'),
                'covered': sum(r['prediction'] is not None for r in group),
                'runtime_errors': sum(r['runtime_error'] for r in group),
                'unparsed_or_abstained': sum(r['prediction'] is None for r in group),
                'pair_correct': sum(vector), 'pair_n': len(ids), 'pair_vector': vector,
                'cap_hits': sum(t['hit_output_cap'] for t in good_traces),
                'mean_input_tokens': avg('input_tokens'), 'mean_generated_tokens_including_special': avg('generated_tokens_including_special'),
                'mean_wall_seconds': avg('wall_seconds'),
                'checker_abstention_reasons': dict(Counter(r['checker']['reason'] for r in group if r['checker'] and r['prediction'] is None))}
        result['models'][model] = {'arms': mr, 'item_ids': ids, 'strata': strata}
        for arm in ['direct', 'reasoned']:
            t = paired_test(mr[arm]['pair_vector'], mr['read']['pair_vector'], strata)
            t.update(model=model, a=arm, b='read_plus_checker')
            result['primary_tests'].append(t)
        result['actual_calls'] += len(calls)
    maximum = 0.
    for rank, t in enumerate(sorted(result['primary_tests'], key=lambda t: t['p'])):
        maximum = max(maximum, min(1., (4 - rank) * t['p']))
        t['holm_p'] = maximum
    return result, predictions


def tables(output, result):
    names = {'qwen7': 'Qwen2.5-VL-7B', 'qwen3': 'Qwen3-VL-8B'}
    labels = {'direct': 'Direct', 'reasoned': 'Reason then answer', 'read': 'Read + rules'}
    lines = [r'\begin{tabular}{llrrrr}', r'\toprule', r'Model & Method & Valid & False & Pair & Covered \\', r'\midrule']
    costs = [r'\begin{tabular}{llrrr}', r'\toprule', r'Model & Method & Mean input & Mean output & Cap hits \\', r'\midrule']
    for model, d in result['models'].items():
        for arm, m in d['arms'].items():
            lines.append(' & '.join([names[model], labels[arm], f"{m['valid_correct']}/128", f"{m['false_correct']}/128",
                f"{m['pair_correct']}/128", f"{m['covered']}/256"]) + r' \\')
            costs.append(' & '.join([names[model], labels[arm], f"{m['mean_input_tokens']:.1f}",
                f"{m['mean_generated_tokens_including_special']:.1f}", str(m['cap_hits'])]) + r' \\')
    for name, contents in [('generated_table.tex', lines), ('generated_costs.tex', costs)]:
        contents.extend([r'\bottomrule', r'\end{tabular}'])
        (output / name).write_bytes(('\n'.join(contents) + '\n').encode())


def analyze(root, output):
    reg = json.loads((root / 'registration.json').read_text())
    if sha(root / 'cases.json') != reg['cases_sha256']:
        raise ValueError('cases drift')
    for name, h in reg['code'].items():
        if sha(root / 'executed_source' / name) != h:
            raise ValueError('executed source drift')
    for name, h in reg['assets'].items():
        if sha(root / name) != h:
            raise ValueError('asset drift')
    cases = json.loads((root / 'cases.json').read_text())
    requests = []
    journals = {'qwen7': [], 'qwen3': []}
    audits = {}
    for shard in (0, 1):
        reqpath = root / f'requests_{shard}.jsonl'
        if sha(reqpath) != reg['requests_sha256'][str(shard)]:
            raise ValueError('request hash drift')
        reqs = read_jsonl(reqpath)
        requests.extend(reqs)
        for model in journals:
            folder = root / f'{model}_{shard}'
            identity = json.loads((folder / 'identity.json').read_text())
            summary = json.loads((folder / 'summary.json').read_text())
            if identity['registration_sha256'] != sha(root / 'registration.json') or identity['requests_sha256'] != sha(reqpath) or identity['config_sha256'] != digest(reg['models'][model]['config']) or identity['code'] != reg['code']:
                raise ValueError('identity drift')
            calls = read_jsonl(folder / 'calls.jsonl')
            if summary['status'] != 'complete' or summary['calls'] != 384 or summary['calls_sha256'] != sha(folder / 'calls.jsonl'):
                raise ValueError('incomplete journal')
            if [c['key'] for c in calls] != [r['key'] for r in reqs] or (folder / '.running').exists() or (folder / 'pending.json').exists():
                raise ValueError('request order or unresolved run')
            journals[model].extend(calls)
            audits[f'{model}_{shard}'] = {'calls_sha256': sha(folder / 'calls.jsonl'), 'identity_sha256': sha(folder / 'identity.json'), 'runtime_errors': summary['errors']}
    result, predictions = calculate(reg, cases, requests, journals)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'raw').mkdir()
    for name in ['cases.json', 'requests_0.jsonl', 'requests_1.jsonl']:
        shutil.copyfile(root / name, output / name)
    design = json.loads(json.dumps(reg))
    for model, meta in design['models'].items():
        meta['config']['name_or_path'] = {'qwen7': 'Qwen/Qwen2.5-VL-7B-Instruct', 'qwen3': 'Qwen/Qwen3-VL-8B-Instruct'}[model]
    design['original_registration_sha256'] = sha(root / 'registration.json')
    write(output / 'study_design.json', design)
    for model, calls in journals.items():
        (output / 'raw' / f'{model}.jsonl').write_bytes((''.join(canonical(c) + '\n' for c in calls)).encode())
    (output / 'predictions.jsonl').write_bytes((''.join(canonical(p) + '\n' for p in predictions)).encode())
    write(output / 'analysis.json', result)
    tables(output, result)
    write(output / 'provenance.json', {'original_registration_sha256': sha(root / 'registration.json'), 'run_audits': audits,
        'assets_verified': len(reg['assets']), 'code_snapshot_files_verified': len(reg['code']),
        'files': {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    print(json.dumps({m: {a: d['pair_correct'] for a, d in v['arms'].items()} for m, v in result['models'].items()}))


def replay(folder):
    prov = json.loads((folder / 'provenance.json').read_text())
    for name, h in prov['files'].items():
        if sha(folder / name) != h:
            raise ValueError('release file drift: ' + name)
    reg = json.loads((folder / 'study_design.json').read_text())
    result, predictions = calculate(reg, json.loads((folder / 'cases.json').read_text()),
        read_jsonl(folder / 'requests_0.jsonl') + read_jsonl(folder / 'requests_1.jsonl'),
        {m: read_jsonl(folder / 'raw' / f'{m}.jsonl') for m in ['qwen7', 'qwen3']})
    if result != json.loads((folder / 'analysis.json').read_text()) or predictions != read_jsonl(folder / 'predictions.jsonl'):
        raise ValueError('numerical replay mismatch')
    print('PASS: 1536 new calls, scene exclusion, prompts, equal output caps, all verdicts, coverage, actual token counts and four corrected paired comparisons replay.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--replay', type=Path)
    a = p.parse_args()
    if a.replay:
        replay(a.replay)
    elif a.root and a.output:
        analyze(a.root, a.output)
    else:
        p.error('provide --replay or --root and --output')
