"""Audit and replay the frozen media and object-swap diagnostic."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cta.verification_workbench import canonical,digest,sha,read_jsonl,exact_read,parse_option
from scripts.channel_study_lib import MEDIA_ARMS,BINDING_ARMS
from scripts.analyze_verification_diagnostic import paired_test


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes((json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode())


def calculate(reg,cases,requests,model_calls):
    expected={r['key']:r for r in requests}
    if len(expected)!=2304 or len(requests)!=2304:raise ValueError('request coverage')
    case_map={(c['study'],c['item_id'],c['state']):c for c in cases}
    if len(case_map)!=384 or len(cases)!=384:raise ValueError('case coverage')
    result={'status':'complete','actual_calls':0,'results':{},'primary_tests':[]}
    for model in ['qwen7','qwen3']:
        calls=model_calls[model];ix={c['key']:c for c in calls}
        if set(ix)!=set(expected) or len(calls)!=2304:raise ValueError('call coverage')
        scores={};unparsed={};errors={}
        for call in calls:
            r=expected[call['key']]
            if call['request']!=r:raise ValueError('call differs from frozen request')
            if call['started_at']<reg['frozen_at']:raise ValueError('inference precedes freeze')
            if (r['study'],r['item_id'],r['state']) not in case_map:raise ValueError('unknown case')
            if not r['image'] and not call['error']:
                trace=call['trace']
                if trace.get('image_supplied') is not False or 'input_ids' not in trace.get('input_keys',[]):raise ValueError('image-free trace missing')
                if any('pixel' in k or 'image_grid' in k or 'video' in k for k in trace['input_keys']):raise ValueError('visual tensor in image-free call')
            if r['arm']=='read':
                correct=bool(not call['error'] and exact_read(call['raw'],r['read_target']));parsed=True
            elif r['arm']=='localize':
                text=call['raw'].strip().upper()
                match=re.fullmatch(r'(?:([LR])|\(([LR])\))[.!]?',text)
                parsed=(match.group(1) or match.group(2)) if match else None
                correct=bool(not call['error'] and parsed==r['gold'])
            else:
                parsed=parse_option(call['raw'],r['option_map'])
                correct=bool(not call['error'] and parsed==r['gold'])
            scores[r['key']]=int(correct)
            unparsed[r['key']]=int(parsed is None)
            errors[r['key']]=int(bool(call['error']))
        model_result={}
        for study,arms,states,n in [('media',MEDIA_ARMS,['record_true','record_false'],128),('binding',BINDING_ARMS,['target_left','target_right'],64)]:
            ids=sorted({c['item_id'] for c in cases if c['study']==study})
            if len(ids)!=n:raise ValueError('item coverage')
            strata=[case_map[study,i,states[0]]['source' if study=='media' else 'family'] for i in ids]
            all_arms=arms+(['read'] if study=='media' else ['localize','text_reasoning'])
            vectors={};metrics={}
            for arm in all_arms:
                keys=[canonical([study,i,s,arm]) for i in ids for s in states]
                vectors[arm]=[int(all(scores[canonical([study,i,s,arm])] for s in states)) for i in ids]
                metrics[arm]={'pair_correct':sum(vectors[arm]),'pair_n':n,'correct':sum(scores[k] for k in keys),'n':2*n,
                    'runtime_errors':sum(errors[k] for k in keys),'unparsed':sum(unparsed[k] for k in keys)}
                if arm not in ['read','localize']:
                    for truth in ['consistent','inconsistent']:
                        group=[k for k in keys if expected[k]['gold']==truth]
                        metrics[arm][truth]={'k':sum(scores[k] for k in group),'n':len(group)}
            controls=['read','oracle_text'] if study=='media' else ['localize','text_reasoning']
            eligible=[int(all(vectors[a][j] for a in controls)) for j in range(n)]
            baseline='scene' if study=='media' else 'none'
            conditioned={'controls':controls,'eligible':sum(eligible),'pair_failures':sum(e and not v for e,v in zip(eligible,vectors[baseline])),
                         'eligible_vector':eligible,'scope':'Descriptive separate-call behavioral conjunction; not internal competence or a superiority test.'}
            model_result[study]={'item_ids':ids,'strata':strata,'metrics':metrics,'pair_vectors':vectors,'control_conditioned':conditioned}
            for contrast in reg[study+'_primary']:
                t=paired_test(vectors[contrast['a']],vectors[contrast['b']],strata)
                t.update(model=model,study=study,**contrast);result['primary_tests'].append(t)
        model_result['runtime_errors']=sum(errors.values());model_result['unparsed']=sum(unparsed.values())
        model_result['actual_calls']=len(calls);result['results'][model]=model_result;result['actual_calls']+=len(calls)
    for study,total in [('media',6),('binding',4)]:
        tests=[t for t in result['primary_tests'] if t['study']==study]
        if len(tests)!=total:raise ValueError('test family')
        maximum=0.
        for rank,t in enumerate(sorted(tests,key=lambda t:t['p'])):
            maximum=max(maximum,min(1.,(total-rank)*t['p']));t['holm_p']=maximum
    return result


def tables(output,result):
    names={'qwen7':'Qwen2.5-VL-7B','qwen3':'Qwen3-VL-8B'}
    labels={'scene':'Scene','record_only':'Record pixels','oracle_scene':'Scene + fields','oracle_record':'Record pixels + fields','oracle_text':'Fields, no image','none':'Image only','fields':'Image + fields','location':'Image + location','both':'Image + fields + location','read':'Exact independent read','localize':'Independent localization','text_reasoning':'Single-record text reasoning'}
    for study in ['media','binding']:
        lines=[r'\begin{tabular}{llrrr}',r'\toprule',r'Model & Input / control & Valid & False & Both states \\',r'\midrule']
        for model,d in result['results'].items():
            for arm,m in d[study]['metrics'].items():
                truth=[f"{m[t]['k']}/{m[t]['n']}" if t in m else '---' for t in ['consistent','inconsistent']]
                lines.append(' & '.join([names[model],labels[arm],*truth,f"{m['pair_correct']}/{m['pair_n']}"])+r' \\')
        lines.extend([r'\bottomrule',r'\end{tabular}'])
        (output/f'generated_{study}_table.tex').write_bytes(('\n'.join(lines)+'\n').encode())
    lines=[r'\begin{tabular}{llrrr}',r'\toprule',r'Model & Contrast ($B-A$) & $\Delta$ (pp) & 95\% CI (pp) & $p_H$ \\',r'\midrule']
    for t in result['primary_tests']:
        label=labels[t['b']]+' -- '+labels[t['a']]
        lines.append(' & '.join([names[t['model']],label,f"{100*t['delta']:+.2f}",f"[{100*t['ci95'][0]:.2f}, {100*t['ci95'][1]:.2f}]",f"{t['holm_p']:.5f}"])+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (output/'generated_tests.tex').write_bytes(('\n'.join(lines)+'\n').encode())


def analyze(root,output):
    reg=json.loads((root/'registration.json').read_text());cases=json.loads((root/'cases.json').read_text())
    if sha(root/'cases.json')!=reg['cases_sha256']:raise ValueError('cases changed')
    for path,h in reg['code'].items():
        if sha(root/'executed_source'/path)!=h:raise ValueError('executed source changed')
    for path,h in reg['assets'].items():
        if sha(root/path)!=h:raise ValueError('image changed')
    # The lower record half must be byte-identical between object-swap states.
    import numpy as np
    from PIL import Image
    ci={(c['study'],c['item_id'],c['state']):c for c in cases}
    for i in sorted({c['item_id'] for c in cases if c['study']=='binding'}):
        a,b=[np.asarray(Image.open(root/ci['binding',i,s]['image']).convert('RGB')) for s in ['target_left','target_right']]
        if not np.array_equal(a[370:],b[370:]):raise ValueError('record pixels changed under object swap')
    requests=[];calls={m:[] for m in ['qwen7','qwen3']};audits={}
    for shard in [0,1]:
        request_file=root/f'requests_{shard}.jsonl'
        if sha(request_file)!=reg['request_hashes'][str(shard)]:raise ValueError('requests changed')
        rows=read_jsonl(request_file);requests.extend(rows)
        for model in calls:
            folder=root/f'{model}_{shard}'
            identity=json.loads((folder/'identity.json').read_text());summary=json.loads((folder/'summary.json').read_text())
            if identity['registration']!=sha(root/'registration.json') or identity['requests']!=sha(request_file) or identity['code']!=reg['code'] or identity['config']!=digest(reg['models'][model]['config']):raise ValueError('run identity mismatch')
            journal=read_jsonl(folder/'calls.jsonl')
            if summary['status']!='complete' or summary['calls']!=1152 or len(journal)!=1152 or summary['calls_sha256']!=sha(folder/'calls.jsonl'):raise ValueError('incomplete or modified journal')
            if [c['key'] for c in journal]!=[r['key'] for r in rows]:raise ValueError('shard order or coverage drift')
            if (folder/'pending.json').exists() or (folder/'.running').exists():raise ValueError('run not closed')
            calls[model].extend(journal)
            audits[f'{model}_{shard}']={'calls_sha256':sha(folder/'calls.jsonl'),'identity_sha256':sha(folder/'identity.json'),'errors':summary['errors']}
    result=calculate(reg,cases,requests,calls)
    output.mkdir(parents=True,exist_ok=False);(output/'raw').mkdir()
    for model,journal in calls.items():
        (output/'raw'/f'{model}.jsonl').write_bytes((''.join(canonical(c)+'\n' for c in journal)).encode())
    for name in ['cases.json','requests_0.jsonl','requests_1.jsonl']:shutil.copyfile(root/name,output/name)
    design=json.loads(json.dumps(reg))
    for model,meta in design['models'].items():meta['config']['name_or_path']={'qwen7':'Qwen/Qwen2.5-VL-7B-Instruct','qwen3':'Qwen/Qwen3-VL-8B-Instruct'}[model]
    design['original_registration_sha256']=sha(root/'registration.json')
    write(output/'study_design.json',design);write(output/'analysis.json',result);tables(output,result)
    files={str(p.relative_to(output)).replace('\\','/'):sha(p) for p in output.rglob('*') if p.is_file()}
    write(output/'provenance.json',{'files':files,'registration_sha256':sha(root/'registration.json'),'run_audits':audits,'original_assets_verified':640,'record_pixel_swap_pairs_verified':64,'analyzer_sha256':sha(Path(__file__)),
        'scope':'Raw replay is portable. Original assets and original registration remain in the research archive; private checkpoint paths are normalized in study_design.'})
    print(json.dumps({k:v for k,v in result.items() if k!='results'}))


def replay(folder):
    provenance=json.loads((folder/'provenance.json').read_text())
    for path,h in provenance['files'].items():
        if sha(folder/path)!=h:raise ValueError('release file changed: '+path)
    reg=json.loads((folder/'study_design.json').read_text());cases=json.loads((folder/'cases.json').read_text())
    requests=read_jsonl(folder/'requests_0.jsonl')+read_jsonl(folder/'requests_1.jsonl')
    calls={m:read_jsonl(folder/'raw'/f'{m}.jsonl') for m in ['qwen7','qwen3']}
    actual=calculate(reg,cases,requests,calls)
    if actual!=json.loads((folder/'analysis.json').read_text()):raise ValueError('derived analysis does not replay')
    print('PASS: 4,608 calls, frozen requests, image-free traces, all metrics, control conjunctions, ten paired tests, bootstrap intervals and separate Holm families. Original pixel audit is separate.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--output',type=Path);p.add_argument('--replay',type=Path)
    a=p.parse_args()
    if a.replay:replay(a.replay)
    elif a.root and a.output:analyze(a.root,a.output)
    else:p.error('use --root and --output, or --replay')
