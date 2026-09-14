"""Pre-inference integrity gate using the separately implemented analyzer."""
import argparse
import collections
import json
from pathlib import Path
import time

from analyze_content import (FAMILIES, check_item_assets, check_reserved_population,
                             require, rows, sha, verify_registration)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    require(not any(base.glob('*/calls.jsonl')), 'Preflight must precede victim calls')
    manifest = base / 'dataset/manifest.jsonl'
    registration = verify_registration(base / 'registration.json', manifest)
    records = rows(manifest)
    require(len(records) == len({row['item_id'] for row in records}) == 128,
            'Expected exactly 128 unique sources')
    require(len({row['source_sha256'] for row in records}) == 128, 'Duplicate source bytes')
    counts = collections.Counter((row['dataset'], row['family']) for row in records)
    require(set(counts) == {(source, family) for source in ('COCO', 'VOC') for family in FAMILIES}
            and set(counts.values()) == {8}, 'Source/family balance changed')
    source_audit = check_reserved_population(records, registration)
    parent = base / 'references/manifest.jsonl'
    originals = {row['item_id']: row for row in rows(parent)}
    require(set(originals) == {row['item_id'] for row in records}, 'Reference population changed')
    for row in records:
        require(row['content_parent_manifest_sha256'] == sha(parent), 'Reference hash changed')
        require({key: value for key, value in row.items()
                 if key not in ('candidates', 'content_parent_manifest_sha256')} == originals[row['item_id']],
                'Immutable reference or task fields changed')
        check_item_assets(row)
    construction = json.loads((base / 'references/build_provenance.json').read_text())
    require(construction['status'] == 'frozen_before_auxiliary_content_or_model_inference'
            and construction['manifest_sha256'] == sha(parent)
            and construction['items'] == 128 and not construction['construction_failures']
            and construction['old64_primitive_tuple_overlap'] == 0
            and construction['new_primitive_tuple_duplicates'] == 0,
            'Reference build was incomplete or not independently sourced')
    require(construction['source_refresh_before_build']['result']['status'] == 'pass',
            'Source-use refresh failed before reference creation')
    report = {'status': 'pass', 'timestamp': time.time(), 'items': len(records),
              'candidate_images': 4 * len(records), 'victim_calls_before_gate': 0,
              'independent_pixel_and_truth_replay': True, 'human_validation': False,
              'registration_sha256': sha(base / 'registration.json'),
              'manifest_sha256': sha(manifest), 'source_audit': source_audit,
              'reference_provenance_sha256': sha(base / 'references/build_provenance.json'),
              'visual_qa_examples': [
                  {'item_id': 'coco-000000067534', 'condition': 'rule_false',
                   'selection': 'first item_id in unit_conversion; inspected before victim inference'},
                  {'item_id': 'coco-000000070229', 'condition': 'rule_false',
                   'selection': 'first item_id in range_threshold; inspected before victim inference'}]}
    with (base / 'preflight.json').open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({'status': 'pass', 'items': 128, 'candidate_images': 512, 'victim_calls': 0}))


if __name__ == '__main__':
    main()
