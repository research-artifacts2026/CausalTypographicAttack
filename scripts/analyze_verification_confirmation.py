"""Audit and export the complete registered 128-scene confirmation.

The primary test and bootstrap are identical to the earlier diagnostic.
Original images are required for this audit; the portable replay is separate.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cta.verification_workbench import (CONDITIONS, GOLD, canonical, digest,
    exact_read, load_packet, parse_option, read_jsonl, score, sha, write_json)
from scripts.analyze_verification_diagnostic import paired_test


def expected_keys(ids,strategies):
    keys={canonical([i,c,s,'decide']) for i in ids for c in CONDITIONS for s in strategies}
    keys.update(canonical([i,c,stage]) for i in ids for c in CONDITIONS
                for stage in ['strategy_read','strategy_draft'])
    keys.update(canonical([i,stage]) for i in ids for stage in ['independent_read','independent_know'])
    return keys


def validate_calls(calls,ids,strategies):
    keys=[c['key'] for c in calls]
    if len(keys)!=len(set(keys)) or set(keys)!=expected_keys(ids,strategies):
        raise ValueError('missing, duplicate, or unexpected call cells')


def analyze(root,output):
    reg=json.loads((root/'registration.json').read_text())
    if reg['items']!=128 or reg['expected_total_calls']!=3584:
        raise ValueError('unexpected registered design')
    rows,packet_ids=[],{}
    for name,meta in reg['packets'].items():
        packet,part=load_packet(root/name)
        if sha(root/name/'packet.json')!=meta['sha256'] or packet['strategies']!=reg['strategies']:
            raise ValueError('packet registration mismatch')
        ids=sorted({r['item_id'] for r in part})
        if ids!=meta['ids'] or len(ids)!=64: raise ValueError('shard selection mismatch')
        packet_ids[name]=ids
        rows.extend(part)
    lookup={(r['item_id'],r['condition']):r for r in rows}
    ids=sorted({r['item_id'] for r in rows})
    if len(ids)!=128 or len(lookup)!=384 or len(rows)!=384:
        raise ValueError('overlapping or incomplete shards')
    if set(ids)&set(reg['excluded_ids']): raise ValueError('diagnostic ID overlap')
    source_hashes={i:sha(Path(lookup[i,'source_absent']['source_path'])) for i in ids}
    frozen={i:h for group in reg['source_hashes'].values() for i,h in group.items()}
    if source_hashes!=frozen or len(set(source_hashes.values()))!=128:
        raise ValueError('source hash mismatch or duplicate')
    if set(source_hashes.values())&set(reg['excluded_source_sha256']):
        raise ValueError('diagnostic source-byte overlap')
    sources=['coco' if i.startswith('coco') else 'voc' for i in ids]
    if {s:sources.count(s) for s in set(sources)}!=reg['source_balance']:
        raise ValueError('source coverage mismatch')
    output.mkdir(parents=True,exist_ok=True)
    (output/'raw').mkdir(exist_ok=True)
    result={'status':'complete','items':128,'actual_calls':0,
        'registration_sha256':sha(root/'registration.json'),'scope':reg['scope'],
        'results':{},'primary_tests':[],'item_ids':ids,'sources':sources,'input_hashes':{},
        'option_maps':{i:lookup[i,'source_absent']['option_map'] for i in ids},
        'read_targets':{i:lookup[i,'record_false']['registered_read_text'] for i in ids},
        'knowledge_options':{i:('B' if lookup[i,'record_false']['knowledge_option_order']=='yes_no' else 'A') for i in ids}}
    for model in reg['models']:
        all_calls,predictions,hashes=[],[],{}
        raw_bytes=[]
        for number in [0,1]:
            name='packet_'+str(number)
            run=root/(model+'_'+str(number))
            calls=read_jsonl(run/'calls.jsonl')
            pred=read_jsonl(run/'predictions.jsonl')
            identity=json.loads((run/'identity.json').read_text())
            original=json.loads((run/'summary.json').read_text())
            validate_calls(calls,packet_ids[name],reg['strategies'])
            if len(calls)!=reg['expected_calls_per_shard']: raise ValueError('incomplete shard')
            if any(c['started_at']<reg['registered_at'] for c in calls):
                raise ValueError('inference before registration')
            if identity['packet']!=reg['packets'][name]['sha256'] or identity['code']!=reg['code']:
                raise ValueError('execution packet/code mismatch')
            if identity['config']!=digest(reg['models'][model]['config']):
                raise ValueError('model configuration mismatch')
            if sha(run/'calls.jsonl')!=original['call_log_sha256']:
                raise ValueError('changed call journal')
            call_map={c['key']:c for c in calls}
            for p in pred:
                i,c,s=p['item_id'],p['condition'],p['strategy']
                row=lookup[i,c]
                call=call_map[canonical([i,c,s,'decide'])]
                if p['raw']!=call['raw'] or p['parsed']!=parse_option(call['raw'],row['option_map']):
                    raise ValueError('decision replay mismatch')
                if call['request']['image_sha256']!=sha(Path(row['image_path'])):
                    raise ValueError('image mismatch')
                if c=='record_false':
                    read=call_map[canonical([i,'independent_read'])]
                    know=call_map[canonical([i,'independent_know'])]
                    if p['read_match']!=exact_read(read['raw'],result['read_targets'][i]):
                        raise ValueError('read replay mismatch')
                    if p['knowledge_correct']!=(know['raw'].strip().upper().strip('().! ')==result['knowledge_options'][i]):
                        raise ValueError('knowledge replay mismatch')
            shard_rows=[r for r in rows if r['item_id'] in packet_ids[name]]
            if score(shard_rows,pred,reg['strategies'])['strategies']!=original['strategies']:
                raise ValueError('stored summary mismatch')
            hashes[str(number)]={f:sha(run/f) for f in ['calls.jsonl','predictions.jsonl','identity.json','model.json']}
            all_calls.extend(calls); predictions.extend(pred)
            raw_bytes.append((run/'calls.jsonl').read_bytes())
        validate_calls(all_calls,ids,reg['strategies'])
        raw=output/'raw'/(model+'.jsonl')
        raw.write_bytes(b''.join(raw_bytes))
        metrics=score(rows,predictions,reg['strategies'])
        indexed={(r['item_id'],r['condition'],r['strategy']):r for r in predictions}
        vectors={s:[int(all(indexed[i,c,s]['parsed']==GOLD[c] for c in ['record_true','record_false'])) for i in ids] for s in reg['strategies']}
        test=paired_test(vectors['self_check'],vectors['read_then_verify'],sources)
        test.update({'model':model,'a':'self_check','b':'read_then_verify'})
        result['primary_tests'].append(test)
        result['results'][model]={'metrics':metrics,'pair_vectors':vectors,'predictions':predictions,
            'actual_calls':len(all_calls),'runtime_errors':sum(bool(c['error']) for c in all_calls)}
        result['actual_calls']+=len(all_calls)
        result['input_hashes'][model]={'calls.jsonl':sha(raw),'shards':hashes}
    maximum=0.
    for rank,t in enumerate(sorted(result['primary_tests'],key=lambda t:t['p'])):
        maximum=max(maximum,min(1.,(2-rank)*t['p'])); t['holm_p']=maximum
    if result['actual_calls']!=reg['expected_total_calls']: raise ValueError('total calls mismatch')
    write_json(output/'analysis.json',result)
    labels={'read_then_verify':'Read then verify','self_check':'Self-check'}
    names={'qwen7':'Qwen2.5-VL-7B','qwen3':'Qwen3-VL-8B'}
    lines=[r'\begin{tabular}{llrrrrrr}',r'\toprule',r'Model & Strategy & Source & Valid & Invalid & Pair & DC-ASR & EOR \\',r'\midrule']
    def count(v): return f"{v['k']}/{v['n']}" if v['n'] else r'--- (0)'
    for model,data in result['results'].items():
        for s,row in data['metrics']['strategies'].items():
            cells=[names[model],labels[s],*[count(row['accuracy'][c]) for c in CONDITIONS],count(row['pair_accuracy']),count(row['dc_asr']),count(row['eor'])]
            lines.append(' & '.join(cells)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (output/'generated_table.tex').write_text('\n'.join(lines)+'\n')
    write_json(output/'table_provenance.json',{'analysis_sha256':sha(output/'analysis.json'),
        'generator_sha256':sha(Path(__file__)),
        'statistics_dependency_sha256':sha(Path(__file__).with_name('analyze_verification_diagnostic.py')),
        'primary_tests':result['primary_tests'],'actual_calls':result['actual_calls']})
    public=json.loads(json.dumps(reg))
    for model,meta in public['models'].items():
        meta['config']['name_or_path']=names[model]
    public['original_private_registration_sha256']=sha(root/'registration.json')
    public['export_note']='Checkpoint filesystem locations replaced with model names; original registration hash retained. No source photographs or weights in this export.'
    write_json(output/'study_design.json',public)
    print(json.dumps({'status':'complete','calls':result['actual_calls'],'primary_tests':[{k:v for k,v in t.items() if not k.endswith('vector')} for t in result['primary_tests']]}))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); analyze(a.root,a.output)
