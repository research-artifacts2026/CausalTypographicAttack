"""Portable raw-response replay of the registered 128-scene confirmation."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cta.verification_workbench import GOLD, canonical, exact_read, parse_option, read_jsonl, score, sha
from scripts.analyze_verification_diagnostic import paired_test
from scripts.analyze_verification_confirmation import validate_calls


def replay(folder):
    data=json.loads((folder/'analysis.json').read_text())
    design=json.loads((folder/'study_design.json').read_text())
    ids=data['item_ids']; rows=[{'item_id':i} for i in ids]
    if data['status']!='complete' or data['actual_calls']!=3584 or len(ids)!=128:
        raise ValueError('incomplete study')
    if set(ids)&set(design['excluded_ids']): raise ValueError('diagnostic overlap')
    tests=[]
    for model in ['qwen7','qwen3']:
        d=data['results'][model]; calls=read_jsonl(folder/'raw'/f'{model}.jsonl')
        if len(calls)!=1792 or sha(folder/'raw'/f'{model}.jsonl')!=data['input_hashes'][model]['calls.jsonl']:
            raise ValueError('call journal mismatch')
        strategies=design['strategies']
        validate_calls(calls,ids,strategies)
        indexed={c['key']:c for c in calls}
        predictions=d['predictions']
        for p in predictions:
            i=p['item_id']
            call=indexed[canonical([i,p['condition'],p['strategy'],'decide'])]
            if p['raw']!=call['raw'] or p['parsed']!=parse_option(call['raw'],data['option_maps'][i]):
                raise ValueError('raw decision mismatch')
            if p['condition']=='record_false':
                read=indexed[canonical([i,'independent_read'])]
                know=indexed[canonical([i,'independent_know'])]
                if p['read_match']!=exact_read(read['raw'],data['read_targets'][i]): raise ValueError('read replay')
                if p['knowledge_correct']!=(know['raw'].strip().upper().strip('().! ')==data['knowledge_options'][i]):
                    raise ValueError('knowledge replay')
        if score(rows,predictions,strategies)!=d['metrics']: raise ValueError('metrics mismatch')
        ix={(p['item_id'],p['condition'],p['strategy']):p for p in predictions}
        vectors={s:[int(all(ix[i,c,s]['parsed']==GOLD[c] for c in ['record_true','record_false'])) for i in ids] for s in strategies}
        if vectors!=d['pair_vectors']: raise ValueError('pair vector mismatch')
        actual=paired_test(vectors['self_check'],vectors['read_then_verify'],data['sources'])
        expected=next(t for t in data['primary_tests'] if t['model']==model)
        for k,v in actual.items():
            if v!=expected[k]: raise ValueError('paired statistic mismatch: '+k)
        tests.append((actual['p'],expected['holm_p']))
    maximum=0.
    for index,(p,expected) in enumerate(sorted(tests)):
        maximum=max(maximum,min(1.,(2-index)*p))
        if not math.isclose(maximum,expected): raise ValueError('Holm mismatch')
    print('PASS: 3,584 calls, disjoint IDs, exact coverage, raw decisions, Read/Know probes, all metrics, paired vectors, bootstrap and Holm correction. Original-pixel audit is separate.')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--evidence',type=Path,required=True)
    replay(p.parse_args().evidence)
