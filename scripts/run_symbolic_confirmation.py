"""Execute a registered shard once; preserve errors and unknown interrupted calls."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cta.verification_workbench import sha, digest, now, write_json, read_jsonl, append
from scripts.symbolic_confirmation_lib import execution_hashes, infer_logged


def run(root, model_name, shard):
    reg = json.loads((root / 'registration.json').read_text())
    if execution_hashes() != reg['code']:
        raise ValueError('execution code changed')
    manifest = root / f'requests_{shard}.jsonl'
    if sha(manifest) != reg['requests_sha256'][str(shard)]:
        raise ValueError('request manifest changed')
    requests = read_jsonl(manifest)
    cfg = reg['models'][model_name]['config']
    folder = root / f'{model_name}_{shard}'
    folder.mkdir(exist_ok=True)
    lock = folder / '.running'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        ident = {'registration_sha256': sha(root / 'registration.json'), 'requests_sha256': sha(manifest),
                 'config_sha256': digest(cfg), 'code': execution_hashes(), 'python': sys.version}
        if (folder / 'identity.json').exists() and json.loads((folder / 'identity.json').read_text()) != ident:
            raise ValueError('run identity changed')
        write_json(folder / 'identity.json', ident)
        if (folder / 'pending.json').exists():
            raise ValueError('unknown interrupted call; manual audit required')
        calls = read_jsonl(folder / 'calls.jsonl')
        if [c['key'] for c in calls] != [r['key'] for r in requests[:len(calls)]]:
            raise ValueError('journal is not an exact request prefix')
        for c, r in zip(calls, requests):
            if c['request'] != r:
                raise ValueError('request drift in existing journal')
        model = None
        for r in requests[len(calls):]:
            picture = (root / r['image']).resolve()
            if not picture.is_relative_to(root.resolve()) or sha(picture) != r['image_sha256']:
                raise ValueError('image drift')
            if model is None:
                from cta.model import build_model_adapter
                model = build_model_adapter(cfg)
                write_json(folder / 'model.json', model.provenance())
            started = now()
            write_json(folder / 'pending.json', {'key': r['key'], 'started_at': started})
            try:
                raw, trace = infer_logged(model, str(picture), r['prompt'], r['tokens'])
                error = None
            except Exception as exc:
                raw, trace, error = '', {}, type(exc).__name__ + ': ' + str(exc)
            c = {'key': r['key'], 'request': r, 'raw': raw, 'trace': trace, 'error': error,
                 'started_at': started, 'finished_at': now()}
            append(folder / 'calls.jsonl', c)
            calls.append(c)
            (folder / 'pending.json').unlink()
            print(f'{len(calls)}/{len(requests)} calls retained', flush=True)
        write_json(folder / 'summary.json', {'status': 'complete', 'calls': len(calls),
            'errors': sum(bool(c['error']) for c in calls), 'calls_sha256': sha(folder / 'calls.jsonl'), 'finished_at': now()})
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', choices=['qwen7', 'qwen3'], required=True)
    p.add_argument('--shard', type=int, choices=[0, 1], required=True)
    a = p.parse_args()
    run(a.root, a.model, a.shard)
