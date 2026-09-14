"""Replay registered strategy comparisons; require complete paired coverage."""
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cta.verification_workbench import (GOLD, CONDITIONS, canonical, load_packet, parse_option,
    read_jsonl, score, sha, write_json, exact_read)


def paired_test(a, b, sources):
    a, b = np.array(a,dtype=int), np.array(b,dtype=int)
    plus, minus = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    n = plus + minus
    p = min(1.,2*sum(math.comb(n,k) for k in range(min(plus,minus)+1))/2**n) if n else 1.
    rng = np.random.default_rng(20260914)
    diff = b-a
    indices = [np.flatnonzero(np.array(sources)==s) for s in sorted(set(sources))]
    boot = np.concatenate([rng.choice(ix,size=(10000,len(ix)),replace=True) for ix in indices],axis=1)
    return {'n':len(a),'a_count':int(a.sum()),'b_count':int(b.sum()),'plus':plus,'minus':minus,
            'delta':float(diff.mean()),'p':p,'ci95':np.quantile(diff[boot].mean(axis=1),[.025,.975]).tolist(),
            'a_vector':a.tolist(),'b_vector':b.tolist()}


def analyze(root, output):
    reg=json.loads((root/'registration.json').read_text())
    packet,rows=load_packet(root/'packet')
    if packet['items'] != 64 or sha(root/'packet/packet.json') != reg['packet_sha256']:
        raise ValueError('registered packet mismatch')
    lookup={(r['item_id'],r['condition']):r for r in rows}
    ids=sorted({r['item_id'] for r in rows})
    sources=['coco' if i.startswith('coco') else 'voc' for i in ids]
    if {s:sources.count(s) for s in set(sources)} != {'coco':32,'voc':32}: raise ValueError('source coverage')
    result={'status':'complete','items':64,'actual_calls':0,'registration_sha256':sha(root/'registration.json'),
            'scope':reg['scope'],'results':{},'primary_tests':[], 'item_ids':ids,'sources':sources,'input_hashes':{},
            'option_maps':{i:lookup[i,'source_absent']['option_map'] for i in ids}}
    for model in ['qwen7','qwen3']:
        run=root/model
        calls=read_jsonl(run/'calls.jsonl'); predictions=read_jsonl(run/'predictions.jsonl')
        identity=json.loads((run/'identity.json').read_text())
        original=json.loads((run/'summary.json').read_text())
        if len(calls) != reg['expected_calls_per_model'] or len({c['key'] for c in calls}) != len(calls):
            raise ValueError('incomplete/duplicate call coverage')
        if any(c['started_at'] < reg['registered_at'] for c in calls): raise ValueError('pre-registration inference')
        if sha(run/'calls.jsonl') != original['call_log_sha256']: raise ValueError('journal changed')
        if identity['packet'] != reg['packet_sha256'] or identity['code'] != reg['code']: raise ValueError('registered code/packet changed')
        call_map={c['key']:c for c in calls}
        for prediction in predictions:
            i,c,s=prediction['item_id'],prediction['condition'],prediction['strategy']
            row=lookup[i,c]
            call=call_map[canonical([i,c,s,'decide'])]
            if prediction['raw'] != call['raw'] or prediction['parsed'] != parse_option(call['raw'],row['option_map']):
                raise ValueError('decision replay mismatch')
            if call['request']['image_sha256'] != sha(Path(row['image_path'])): raise ValueError('image mismatch')
            if c == 'record_false':
                reading=call_map[canonical([i,'independent_read'])]
                knowledge=call_map[canonical([i,'independent_know'])]
                expected='B' if row['knowledge_option_order']=='yes_no' else 'A'
                if prediction['read_match'] != exact_read(reading['raw'],row['registered_read_text']): raise ValueError('read replay')
                if prediction['knowledge_correct'] != (knowledge['raw'].strip().upper().strip('().! ')==expected): raise ValueError('know replay')
        replay=score(rows,predictions,reg['strategies'])
        if replay['strategies'] != original['strategies']: raise ValueError('summary replay mismatch')
        indexed={(r['item_id'],r['condition'],r['strategy']):r for r in predictions}
        vectors={s:[int(all(indexed[i,c,s]['parsed']==GOLD[c] for c in ['record_true','record_false'])) for i in ids] for s in reg['strategies']}
        t=paired_test(vectors['self_check'],vectors['read_then_verify'],sources)
        t.update({'model':model,'a':'self_check','b':'read_then_verify'})
        result['primary_tests'].append(t)
        result['results'][model]={'metrics':replay,'pair_vectors':vectors,'predictions':predictions,
                                  'actual_calls':len(calls),'runtime_errors':sum(bool(c['error']) for c in calls)}
        result['actual_calls']+=len(calls)
        result['input_hashes'][model]={name:sha(run/name) for name in ['calls.jsonl','predictions.jsonl','identity.json','model.json']}
    maximum=0.
    for rank,t in enumerate(sorted(result['primary_tests'],key=lambda t:t['p'])):
        maximum=max(maximum,min(1.,(2-rank)*t['p'])); t['holm_p']=maximum
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'analysis.json',result)
    labels={'direct':'Direct','explicit_rule':'Rule-guided','read_then_verify':'Read then verify','self_check':'Self-check'}
    model_names={'qwen7':'Qwen2.5-VL-7B','qwen3':'Qwen3-VL-8B'}
    lines=[r'\begin{tabular}{llrrrrrr}',r'\toprule',r'Model & Strategy & Source & Valid & Invalid & Pair & DC-ASR & EOR \\',r'\midrule']
    def count(v): return f"{v['k']}/{v['n']}" if v['n'] else r'--- (0)'
    for model,data in result['results'].items():
        for s,row in data['metrics']['strategies'].items():
            cells=[model_names[model],labels[s],*[count(row['accuracy'][c]) for c in CONDITIONS],count(row['pair_accuracy']),count(row['dc_asr']),count(row['eor'])]
            lines.append(' & '.join(cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (output/'generated_table.tex').write_text('\n'.join(lines)+'\n')
    write_json(output/'table_provenance.json',{'analysis_sha256':sha(output/'analysis.json'),'generator_sha256':sha(Path(__file__)),
                                              'primary_tests':result['primary_tests'],'actual_calls':result['actual_calls']})
    print(json.dumps({'status':'complete','calls':result['actual_calls'],'primary_tests':[{k:v for k,v in t.items() if not k.endswith('vector')} for t in result['primary_tests']]}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); analyze(args.root,args.output)
