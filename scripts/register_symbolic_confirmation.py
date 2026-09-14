"""Freeze a scene-disjoint, equal-output-cap symbolic-checker confirmation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cta.verification_workbench import read_jsonl, load_packet, sha, canonical, now, write_json
from cta.transcribed_record_checker import check_record
from scripts.register_verification_confirmation import choose_disjoint
from scripts.symbolic_confirmation_lib import ARMS, prompts, execution_hashes


def build(a):
    previous = json.loads((a.retrospective / 'cases.json').read_text())
    previous = [c for c in previous if c['study'] == 'media']
    _, diagnostic = load_packet(a.diagnostic)
    excluded_ids = {c['item_id'] for c in previous} | {c['item_id'] for c in diagnostic}
    excluded_hashes = {c['source_sha256'] for c in previous} | {sha(Path(c['source_path'])) for c in diagnostic}
    rows = []
    source_hashes = {}
    for source, path in [('coco', a.coco), ('voc', a.voc)]:
        chosen, hashes = choose_disjoint(read_jsonl(path), excluded_ids, excluded_hashes, 64)
        rows.extend(chosen)
        source_hashes.update(hashes)
    if len(source_hashes) != 128 or len(set(source_hashes.values())) != 128:
        raise ValueError('source identity coverage')
    ids = sorted(source_hashes)
    lookup = {(r['item_id'], r['condition']): r for r in rows}
    # Reject the entire registration on nominal-rule disagreement, never filter a case.
    for i in ids:
        for state, gold in [('record_true', 'consistent'), ('record_false', 'inconsistent')]:
            r = lookup[i, state]
            if check_record(r['registered_read_text'], r['record']['assumption'])['prediction'] != gold:
                raise ValueError('nominal-rule sanity mismatch: ' + i + ' ' + state)
            if sha(Path(r['image_path'])) != r['image_sha256']:
                raise ValueError('original rendered image mismatch')
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'assets').mkdir()
    cases, requests = [], [[], []]
    for j, i in enumerate(ids):
        for state, gold in [('record_true', 'consistent'), ('record_false', 'inconsistent')]:
            r = lookup[i, state]
            original = Path(r['image_path'])
            name = 'assets/' + sha(original) + original.suffix.lower()
            if not (a.output / name).exists():
                shutil.copyfile(original, a.output / name)
            assumption, mapping = r['record']['assumption'], r['option_map']
            ps = prompts(assumption, mapping)
            c = {'item_id': i, 'state': state, 'gold': gold, 'family': r['family'],
                 'source': 'coco' if i.startswith('coco') else 'voc', 'source_sha256': source_hashes[i],
                 'image': name, 'image_sha256': sha(original), 'assumption': assumption,
                 'fields': r['registered_read_text'], 'option_map': mapping}
            cases.append(c)
            # Rotate call order deterministically; no outcome-dependent ordering.
            arms = ARMS[j % 3:] + ARMS[:j % 3]
            for arm in arms:
                requests[j % 2].append({'key': canonical([i, state, arm]), 'item_id': i, 'state': state,
                    'arm': arm, 'image': name, 'image_sha256': sha(original), 'prompt': ps[arm], 'tokens': 384})
    write_json(a.output / 'cases.json', cases)
    hashes = {}
    for shard in (0, 1):
        path = a.output / f'requests_{shard}.jsonl'
        path.write_bytes((''.join(canonical(r) + '\n' for r in requests[shard])).encode())
        hashes[str(shard)] = sha(path)
    old_registration = json.loads((a.model_registration).read_text())
    models = old_registration['models']
    checkpoint_hashes = {}
    for model, meta in models.items():
        cfg = meta['config']
        cfg['max_new_tokens'] = 384
        cfg['do_sample'] = False
        cfg['device'] = 'cuda:0'
        checkpoint = Path(cfg['name_or_path'])
        wanted = [p for p in checkpoint.iterdir() if p.is_file() and (p.suffix == '.safetensors' or p.suffix == '.json' or p.name in ('merges.txt', 'tokenizer.model'))]
        checkpoint_hashes[model] = {p.name: sha(p) for p in sorted(wanted)}
    code = execution_hashes()
    root = Path(__file__).resolve().parents[1]
    for name in code:
        target = a.output / 'executed_source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    strata = Counter(c['source'] + '/' + c['family'] for c in cases if c['state'] == 'record_true')
    if len(strata) != 16 or set(strata.values()) != {8}:
        raise ValueError('source/family balance')
    write_json(a.output / 'registration.json', {
        'frozen_at': now(), 'items': 128, 'cases': 256, 'models': models,
        'checkpoint_files_sha256_before_inference': checkpoint_hashes, 'code': code,
        'cases_sha256': sha(a.output / 'cases.json'), 'requests_sha256': hashes,
        'assets': {p.relative_to(a.output).as_posix(): sha(p) for p in sorted((a.output / 'assets').iterdir())},
        'source_family_balance': dict(strata), 'excluded_item_ids': sorted(excluded_ids),
        'excluded_source_sha256': sorted(excluded_hashes),
        'retrospective_cases_sha256': sha(a.retrospective / 'cases.json'),
        'previous_diagnostic_packet_sha256': sha(a.diagnostic / 'packet.json'),
        'arms': ARMS, 'shards': 2, 'calls_per_shard': 384, 'calls_per_model': 768, 'total_calls': 1536,
        'selection': '64 per source, eight per family; stable ID-hash selection after excluding both earlier studies by ID and original source SHA256. No model outcome input.',
        'budget': 'Each arm has one model call per record and a 384 output-token upper bound. Actual generated/input tokens and wall time are recorded, not assumed equal. Checker uses additional fixed CPU rules.',
        'primary': 'All-item valid-invalid pair accuracy; symbolic checker versus direct, and versus reasoned, per model. Exact two-sided McNemar; Holm across all four contrasts.',
        'uncertainty': '10000 paired source-and-family-stratified bootstrap draws; seed 20260914. Marginal percentile 95% intervals.',
        'secondary': 'Valid/false accuracy, coverage, abstention, parsing failures, runtime errors, token usage and latency; descriptive.',
        'checker': 'Exact previously published checker bytes retained. Inputs only raw read text and public assumption; no gold, ID, family label, target fields or generator parameters.',
        'stopping': 'All registered calls once. Preserve errors, cap hits and unparsed outputs. No silent retry, outcome-based selection, early significance stopping or prompt/checker edits.',
        'scope': 'Prospective scene-disjoint confirmation relative to the two prior diagnostics. Archived benchmark scenes and known record schemas, not globally unseen sources, novel tasks, or independent human gold certification.'})
    print(json.dumps({'items': 128, 'calls': 1536, 'registration_sha256': sha(a.output / 'registration.json')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('coco', 'voc', 'retrospective', 'diagnostic', 'model-registration', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    build(p.parse_args())
