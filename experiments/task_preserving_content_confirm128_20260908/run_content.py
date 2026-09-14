"""Fresh fixed-question inference for all registered content cells; no search."""
import argparse
import copy
import json
import time
from pathlib import Path
from search import Journal, RealAdapter, load_config, parse_choice, file_hash, write_or_verify, digest

CONDITIONS = ('clean', 'simple_false', 'rule_false', 'simple_true', 'rule_true')


def verify_registration(base):
    registration = json.loads((base/'registration.json').read_text())
    if file_hash(base/'dataset/manifest.jsonl') != registration['manifest_sha256']:
        raise ValueError('Registered manifest changed')
    for filename, sha in registration['code_sha256'].items():
        if file_hash(filename) != sha:
            raise ValueError('Registered runtime changed: '+str(filename))
    for model, cfg in registration['config_files'].items():
        if file_hash(cfg['path']) != cfg['sha256']:
            raise ValueError('Registered model config changed: '+model)
    for key in ('source_reservation', 'prior_use_registry'):
        value = registration[key]
        if file_hash(value['path']) != value['sha256']:
            raise ValueError('Source audit artifact changed: '+key)
    for value in registration['reference_artifacts'].values():
        if file_hash(value['path']) != value['sha256']:
            raise ValueError('Reference construction artifact changed: '+value['path'])
    return registration


def freeze(root, base):
    here = Path(__file__).resolve().parent
    if not (here/'analyze_content.py').exists():
        raise ValueError('Independent analyzer must exist before registration')
    codes = {str(p): file_hash(p) for p in sorted(here.glob('*.py'))}
    codes[str(here/'PROTOCOL.md')] = file_hash(here/'PROTOCOL.md')
    codes[str(root/'cta/model.py')] = file_hash(root/'cta/model.py')
    configs = {m: {'path': str(base/'configs'/(m+'.json')), 'sha256': file_hash(base/'configs'/(m+'.json'))}
               for m in ('qwen7', 'qwen3vl8')}
    manifest = base/'dataset/manifest.jsonl'
    audit = root/'runs/experiments_gap_20260908_followup/source_audit'
    reservation = audit/'prospective_source_reservation_n128.jsonl'
    registry = audit/'used_identity_registry.json'
    write_or_verify(base/'registration.json', {'manifest_sha256': file_hash(manifest),
        'reference_artifacts': {name: {'path': str(base/'references'/name),
                                      'sha256': file_hash(base/'references'/name)}
                                for name in ('manifest.jsonl', 'build_provenance.json')},
        'source_reservation': {'path': str(reservation), 'sha256': file_hash(reservation)},
        'prior_use_registry': {'path': str(registry), 'sha256': file_hash(registry)},
        'code_sha256': codes, 'config_files': configs, 'items': 128,
        'conditions': list(CONDITIONS), 'fresh_calls_expected': 1280,
        'timestamp': time.time(), 'primary_tests': 'two rule_false-minus-simple_false exact paired tests; Holm',
        'endpoint': 'same clean-correct denominator; no read or true-twin eligibility gate in primary metric',
        'scope': 'prospectively reserved sources within audited scope and new numeric instances; fixed-reference content confirmation, not original three-state protocol or SOTA'})


def run(root, base, model):
    registration = verify_registration(base)
    items = [json.loads(x) for x in (base/'dataset/manifest.jsonl').read_text().splitlines() if x.strip()]
    if len(items) != 128 or len({i['item_id'] for i in items}) != 128:
        raise ValueError('Incomplete item population')
    cfgfile = Path(registration['config_files'][model]['path'])
    cfg = load_config(cfgfile, 'cuda:0', planner=False)
    out = base/model
    provenance = {'model': model, 'registration_sha256': file_hash(base/'registration.json'),
        'manifest_sha256': registration['manifest_sha256'], 'config_sha256': file_hash(cfgfile),
        'effective_config': cfg, 'seed': 20260908, 'max_new_tokens': 96,
        'fresh_clean': True, 'shared_question_across_all_cells': True}
    write_or_verify(out/'provenance.json', provenance)
    victim = RealAdapter(cfg, root)
    write_or_verify(out/'adapter.json', victim.provenance())
    victim.set_call_seed(20260908)
    journal = Journal(out/'calls.jsonl')
    for item in items:
        for condition in CONDITIONS:
            path = item['reference_image_path'] if condition == 'clean' else item['candidates'][condition]['image_path']
            expected = item['reference_image_sha256'] if condition == 'clean' else item['candidates'][condition]['image_sha256']
            if file_hash(path) != expected:
                raise ValueError('Candidate/reference bytes changed')
            metadata = {'item_id': item['item_id'], 'condition': condition, 'kind': 'decision',
                'phase': 'clean' if condition == 'clean' else 'attack', 'family': item['family'],
                'dataset': item['dataset'], 'question_sha256': digest(item['question']),
                'gold': 'correct', 'target': 'target'}
            journal.query(victim, item['item_id']+':'+condition, metadata, path, item['question'], 96,
                          parser=lambda raw: parse_choice(raw, item))
        print(json.dumps({'model': model, 'item_completed': item['item_id']}), flush=True)
    failures = [v['call_id'] for v in journal.finished.values() if v['status'] != 'ok']
    expected_ids = {item['item_id']+':'+c for item in items for c in CONDITIONS}
    if set(journal.started) != expected_ids or set(journal.finished) != expected_ids:
        raise ValueError('Incomplete or unregistered inference calls')
    write_or_verify(out/'complete.json', {'status': 'incomplete_runtime' if failures else 'complete',
        'expected_calls': 640, 'finished_calls': len(journal.finished), 'failures': failures,
        'calls_sha256': file_hash(out/'calls.jsonl'), 'provenance_sha256': file_hash(out/'provenance.json')})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('freeze', 'run', 'verify'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--model', choices=('qwen7', 'qwen3vl8'))
    a = p.parse_args()
    a.root, a.base = a.root.resolve(), a.base.resolve()
    if a.action == 'freeze': freeze(a.root, a.base)
    elif a.action == 'verify': print(json.dumps({'status': 'registered_files_match', 'files': len(verify_registration(a.base)['code_sha256'])}))
    else:
        if a.model is None: p.error('--model required for inference')
        run(a.root, a.base, a.model)
