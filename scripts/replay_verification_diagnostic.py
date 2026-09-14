"""Portable response/statistics replay; does not claim access to image pixels."""
import argparse
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cta.verification_workbench import GOLD, canonical, parse_option, read_jsonl, score, sha
from scripts.analyze_verification_diagnostic import paired_test


def replay(folder):
    data=json.loads((folder/'analysis.json').read_text())
    ids=data['item_ids']; rows=[{'item_id':i} for i in ids]
    if data['status']!='complete' or data['actual_calls']!=2560 or len(ids)!=64: raise ValueError('incomplete study')
    tests=[]
    for model in ['qwen7','qwen3']:
        d=data['results'][model]; calls=read_jsonl(folder/'raw'/f'{model}.jsonl')
        if len(calls)!=1280 or sha(folder/'raw'/f'{model}.jsonl')!=data['input_hashes'][model]['calls.jsonl']: raise ValueError('call journal mismatch')
        indexed={c['key']:c for c in calls}
        predictions=d['predictions']
        for p in predictions:
            call=indexed[canonical([p['item_id'],p['condition'],p['strategy'],'decide'])]
            if p['raw']!=call['raw'] or p['parsed']!=parse_option(call['raw'],data['option_maps'][p['item_id']]): raise ValueError('raw decision mismatch')
        strategies=list(d['metrics']['strategies'])
        if score(rows,predictions,strategies)!=d['metrics']: raise ValueError('metrics mismatch')
        ix={(p['item_id'],p['condition'],p['strategy']):p for p in predictions}
        vectors={s:[int(all(ix[i,c,s]['parsed']==GOLD[c] for c in ['record_true','record_false'])) for i in ids] for s in strategies}
        if vectors!=d['pair_vectors']: raise ValueError('pair vector mismatch')
        actual=paired_test(vectors['self_check'],vectors['read_then_verify'],data['sources'])
        expected=next(t for t in data['primary_tests'] if t['model']==model)
        for k,v in actual.items():
            if v!=expected[k]: raise ValueError(f'paired statistic mismatch: {k}')
        tests.append((actual['p'],expected['holm_p']))
    maximum=0.
    for index,(p,expected) in enumerate(sorted(tests)):
        maximum=max(maximum,min(1.,(2-index)*p))
        if not math.isclose(maximum,expected): raise ValueError('Holm mismatch')
    print('PASS: 2,560 raw calls, option maps, all metrics, paired vectors, stratified intervals and Holm tests. Image-pixel replay is separate.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--evidence',type=Path,required=True)
    replay(parser.parse_args().evidence)
