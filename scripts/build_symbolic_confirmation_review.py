"""Sample a blinded human-review packet without consulting model outcomes."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def build(root, output):
    cases = json.loads((root / 'cases.json').read_text())
    reg = json.loads((root / 'registration.json').read_text())
    if hashlib.sha256((root / 'cases.json').read_bytes()).hexdigest() != reg['cases_sha256']:
        raise ValueError('cases changed')
    buckets = {}
    for c in cases:
        buckets.setdefault((c['source'], c['family']), set()).add(c['item_id'])
    if len(buckets) != 16:
        raise ValueError('expected sixteen source/family strata')
    selected = {i for ids in buckets.values() for i in sorted(ids, key=lambda i: h('symbolic-human-sample-v1:' + i))[:2]}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'images').mkdir()
    rows, key = [], []
    for c in cases:
        if c['item_id'] not in selected:
            continue
        rid = 'SC-' + h('symbolic-human-case-v1:' + c['item_id'] + c['state'])[:12]
        source = (root / c['image']).resolve()
        if not source.is_relative_to(root.resolve()) or hashlib.sha256(source.read_bytes()).hexdigest() != c['image_sha256']:
            raise ValueError('image drift')
        name = 'images/' + rid + source.suffix
        shutil.copyfile(source, output / name)
        rows.append({'review_id': rid, 'image': name, 'task': 'Assess the internal consistency of the displayed record.',
                     'prompt': 'Assumptions: ' + c['assumption']})
        key.append({'review_id': rid, 'item_id': c['item_id'], 'state': c['state'],
                    'gold': c['gold'], 'source': c['source'], 'family': c['family']})
    rows.sort(key=lambda r: r['review_id'])
    key.sort(key=lambda r: r['review_id'])
    if len(rows) != 64:
        raise ValueError('expected 64 cases')
    (output / 'cases.json').write_bytes((json.dumps(rows, indent=2) + '\n').encode())
    output.with_name(output.name + '_PRIVATE_KEY.json').write_bytes((json.dumps(key, indent=2) + '\n').encode())
    columns = ['reviewer_id', 'review_id', 'judgment', 'assumptions_sufficient', 'readability', 'target_visible', 'notes']
    with (output / 'annotations_blank.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({'review_id': r['review_id']} for r in rows)
    # Reuse only the existing offline UI literal, not its selection or labels.
    tree = ast.parse(Path(__file__).with_name('build_blinded_channel_review.py').read_text(encoding='utf-8'))
    template = next(ast.literal_eval(n.value) for n in ast.walk(tree)
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'template' for t in n.targets))
    template = template.replace('record-review-v1', 'symbolic-confirmation-review-v1')
    (output / 'review.html').write_text(template.replace('__CASES__', json.dumps(rows).replace('<', '\\u003c')), encoding='utf-8', newline='\n')
    (output / 'README.txt').write_text(
        'Unzip and open review.html. Each reviewer should use a separate copy/browser profile and a pseudonym. '
        'Review all 64 cases independently and download the CSV. All labels start blank. '
        'Selection takes two items from each of sixteen source/family strata, retains both record states, '
        'and shuffles by a fixed hash without reading any model responses. No reference labels are included. '
        'This is a sample of the new confirmation population, not a full human gold audit. '
        'Disagreements, ambiguous assumptions or systematic family problems require independent adjudication and expanded review.\n',
        encoding='utf-8', newline='\n')
    with zipfile.ZipFile(output.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(output).as_posix())
    print(json.dumps({'cases': len(rows), 'sampled_items': len(selected), 'zip_sha256': hashlib.sha256(output.with_suffix('.zip').read_bytes()).hexdigest()}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    build(a.root, a.output)
