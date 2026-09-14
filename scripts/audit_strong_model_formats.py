"""Descriptive post-run audit; does not change frozen parsers or primary scores."""
import argparse,hashlib,json
from pathlib import Path


def audit(folder):
    load=lambda n:[json.loads(s) for s in (folder/n).read_text(encoding='utf-8').splitlines()]
    predictions=load('predictions.jsonl')
    assert len(predictions)==1280
    calls={(r['request']['item_id'],r['request']['state'],r['request']['arm']):r for r in load('calls.jsonl')}
    cases={(c['item_id'],c['state']):c for c in json.loads((folder/'cases.json').read_text(encoding='utf-8'))}
    report={'scope':'Post-run descriptive format audit and lexicographically first illustrative missing-range example; primary scores unchanged.','arms':{}}
    for arm in ['direct','reasoned','read','reasoned_long','thinking']:
        group=[p for p in predictions if p['arm']==arm]
        report['arms'][arm]={'unparsed_at_cap':sum(p['prediction'] is None and p['trace'].get('cap_hit',False) for p in group),
                            'unparsed_not_at_cap':sum(p['prediction'] is None and not p['trace'].get('cap_hit',False) for p in group)}
    for arm in ['reasoned','reasoned_long','thinking']:
        assert report['arms'][arm]['unparsed_not_at_cap']==0
    group=[p for p in predictions if p['arm']=='read']
    report['covered_read_disagreements']=sum(p['prediction'] is not None and not p['correct'] for p in group)
    assert report['covered_read_disagreements']==0
    for p in sorted(group,key=lambda p:(p['item_id'],p['state'])):
        c=cases[p['item_id'],p['state']];call=calls[p['item_id'],p['state'],'read']
        if p['prediction'] is None and 'SAFE RANGE' in c['fields'] and 'SAFE RANGE' not in call['raw'].upper():
            report['illustrative_read_omission']={'item_id':p['item_id'],'state':p['state'],'nominal_fields':c['fields'],'raw_transcription':call['raw'],'checker_reason':p['checker']['reason']}
            break
    report['input_hashes']={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ['predictions.jsonl','calls.jsonl','cases.json']}
    (folder/'format_audit.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report['arms']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);audit(p.parse_args().evidence)
