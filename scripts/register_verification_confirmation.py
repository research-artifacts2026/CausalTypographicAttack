"""Register a disjoint-scene, fixed-prompt confirmation before any inference.

These are archived benchmark scenes, not globally unseen images. Exclusion is
both by item ID and exact original source bytes against the earlier diagnostic.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cta.verification_workbench import (freeze_packet, load_packet, read_jsonl,
    select_items, sha, now, source_code_hashes, write_json)


def choose_disjoint(rows, excluded_ids, excluded_hashes, count=64):
    source_rows = [r for r in rows if r['condition'] == 'source_absent']
    hashes = {r['item_id']: sha(Path(r['source_path'])) for r in source_rows}
    allowed = {i for i,h in hashes.items() if i not in excluded_ids and h not in excluded_hashes}
    selected = select_items([r for r in rows if r['item_id'] in allowed], count)
    clean = [r for r in selected if r['condition'] == 'source_absent']
    if len({hashes[r['item_id']] for r in clean}) != count:
        raise ValueError('duplicate source bytes within confirmation')
    if set(Counter(r['family'] for r in clean).values()) != {8}:
        raise ValueError('confirmation must have eight scenes per source/family')
    return selected, {r['item_id']: hashes[r['item_id']] for r in clean}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['coco','voc','diagnostic','qwen7','qwen3','output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    _, previous = load_packet(a.diagnostic)
    excluded_ids = {r['item_id'] for r in previous}
    excluded_hashes = {sha(Path(r['source_path'])) for r in previous}
    chosen, selection = {}, {}
    for name, path in [('coco',a.coco),('voc',a.voc)]:
        chosen[name], selection[name] = choose_disjoint(read_jsonl(path), excluded_ids, excluded_hashes)
    all_hashes = [h for group in selection.values() for h in group.values()]
    if len(set(all_hashes)) != 128:
        raise ValueError('duplicate source bytes across sources')
    a.output.mkdir(parents=True, exist_ok=False)
    strategies = ['read_then_verify','self_check']
    shards = [[],[]]
    for rows in chosen.values():
        first = select_items(rows,32)
        first_ids = {r['item_id'] for r in first}
        shards[0].extend(first)
        shards[1].extend(r for r in rows if r['item_id'] not in first_ids)
    packets = {}
    for index, rows in enumerate(shards):
        name = 'packet_'+str(index)
        packet = freeze_packet(rows,a.output/name,origin='Fixed-prompt confirmation disjoint from the 64-scene mitigation diagnostic; archived benchmark images',strategies=strategies)
        packets[name] = {'sha256':sha(packet/'packet.json'),'items':64,
                         'ids':sorted({r['item_id'] for r in rows})}
    configs = {}
    for name,path in [('qwen7',a.qwen7),('qwen3',a.qwen3)]:
        cfg = yaml.safe_load(path.read_text())['model']
        checkpoint = Path(cfg['name_or_path'])
        configs[name] = {'config':cfg,'checkpoint_config_sha256':sha(checkpoint/'config.json'),
                         'checkpoint_revision':checkpoint.name if checkpoint.parent.name == 'snapshots' else None}
        (a.output/(name+'.local.yaml')).write_text(yaml.safe_dump({'model':cfg}))
    code = source_code_hashes()
    project = Path(__file__).resolve().parents[1]
    for name in code:
        dst = a.output/'executed_source'/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(project/name,dst)
    write_json(a.output/'registration.json',{
        'registered_at':now(),'items':128,'source_balance':{'coco':64,'voc':64},
        'selection':'Exclude prior diagnostic by item ID and exact source SHA256; stable item-ID SHA256 order round-robin over eight families; eight scenes per source/family. No model-output input.',
        'source_hashes':selection,'excluded_ids':sorted(excluded_ids),'excluded_source_sha256':sorted(excluded_hashes),
        'diagnostic_packet_sha256':sha(a.diagnostic/'packet.json'),'packets':packets,
        'strategies':strategies,'models':configs,'code':code,
        'registration_script_sha256':sha(Path(__file__)),
        'expected_calls_per_shard':896,'expected_calls_per_model':1792,'expected_total_calls':3584,
        'primary_endpoint':'All-scene valid-invalid pair accuracy, read_then_verify minus self_check, separately for both models',
        'primary_test':'Two-sided exact paired McNemar; Holm across the two registered model contrasts',
        'ci':'10000 paired source-stratified bootstrap resamples, seed 20260914',
        'secondary_endpoints':'All-state accuracy, balanced true/false accuracy, control coverage, DC-ASR and EOR with denominators; descriptive only',
        'budget':'Both arms: 384 draft/transcription + 96 final output-token cap per state, greedy decoding. Six calls each per triplet plus two shared independent Read/Know calls. Actual lengths are not matched.',
        'stopping':'Every registered cell once; retain errors and unparsed answers; no outcome-dependent stopping, selection, prompt changes or silent retries',
        'scope':'Scene-disjoint confirmation of an exploratory mitigation contrast. Archived images were used in earlier benchmark studies; this is not globally unseen-source transfer, a mechanistic identification, or an attack-baseline superiority experiment.'})
    print(json.dumps({'items':128,'expected_calls':3584,'registration_sha256':sha(a.output/'registration.json')}))


if __name__ == '__main__': main()
