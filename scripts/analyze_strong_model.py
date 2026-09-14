"""Replay the complete frozen larger-model comparison, including null outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cta.verification_workbench import sha, read_jsonl, canonical, parse_option
from cta.transcribed_record_checker import check_record
from scripts.symbolic_confirmation_lib import parse_reasoned
from scripts.strong_model_lib import ARMS, LABELS, requests_for, write
from scripts.analyze_verification_diagnostic import paired_test


def calculate(reg,cases,requests,calls):
    ci={(c['item_id'],c['state']):c for c in cases}
    ids=sorted({c['item_id'] for c in cases})
    if len(cases)!=256 or len(ci)!=256 or len(ids)!=128 or requests!=requests_for(cases):
        raise ValueError('case, prompt or budget drift')
    expected={r['key']:r for r in requests}
    if len(calls)!=1280 or len({c['key'] for c in calls})!=1280 or {c['key'] for c in calls}!=set(expected):
        raise ValueError('incomplete or duplicate calls')
    rows=[]
    for call in calls:
        r=expected[call['key']]; c=ci[r['item_id'],r['state']]
        if call['request']!=r or call['started_at']<reg['frozen_at']:
            raise ValueError('request or freeze-time drift')
        checked=None;pred=None
        if not call['error']:
            response=call['response'];choice=response['choices'][0];msg=choice['message'];usage=response['usage']
            if response['model']!='qwen35-27b-cta' or (msg.get('content') or '')!=call['raw']:
                raise ValueError('raw response mismatch')
            if (msg.get('reasoning') or msg.get('reasoning_content') or '')!=call['reasoning']:
                raise ValueError('reasoning response mismatch')
            t=call['trace']
            if not 0<=usage['completion_tokens']<=r['tokens'] or t['output_tokens']!=usage['completion_tokens'] or t['input_tokens']!=usage['prompt_tokens']:
                raise ValueError('token trace mismatch')
            if t['finish_reason']!=choice['finish_reason'] or t['cap_hit']!=(choice['finish_reason']=='length'):
                raise ValueError('finish trace mismatch')
            if r['arm']=='read':
                checked=check_record(call['raw'],c['assumption']);pred=checked['prediction']
            elif r['arm']=='direct': pred=parse_option(call['raw'],c['option_map'])
            else: pred=parse_reasoned(call['raw'],c['option_map'])
        rows.append({'item_id':c['item_id'],'state':c['state'],'arm':r['arm'],'gold':c['gold'],
                     'prediction':pred,'correct':int(pred==c['gold']),'runtime_error':bool(call['error']),
                     'checker':checked,'trace':call['trace']})
    rows.sort(key=lambda x:(x['item_id'],x['state'],x['arm']))
    result={'status':'complete','actual_calls':1280,'items':128,'model_id':reg['model_id'],'arms':{},'registered_tests':[]}
    strata=[ci[i,'record_true']['source']+'/'+ci[i,'record_true']['family'] for i in ids]
    for arm in ARMS:
        group=[r for r in rows if r['arm']==arm];ix={(r['item_id'],r['state']):r for r in group}
        vector=[int(all(ix[i,s]['correct'] for s in ['record_true','record_false'])) for i in ids]
        optimistic=[int(all(ix[i,s]['correct'] or (ix[i,s]['prediction'] is None and not ix[i,s]['runtime_error']) for s in ['record_true','record_false'])) for i in ids]
        good=[r['trace'] for r in group if not r['runtime_error']]
        result['arms'][arm]={'n':256,'valid_correct':sum(r['correct'] for r in group if r['state']=='record_true'),
            'false_correct':sum(r['correct'] for r in group if r['state']=='record_false'),
            'pair_correct':sum(vector),'pair_n':128,'pair_vector':vector,
            'covered':sum(r['prediction'] is not None for r in group),'runtime_errors':sum(r['runtime_error'] for r in group),
            'unparsed_or_abstained':sum(r['prediction'] is None for r in group),'optimistic_pair_correct':sum(optimistic),
            'cap_hits':sum(t['cap_hit'] for t in good),
            'mean_input_tokens':sum(t['input_tokens'] for t in good)/len(good) if good else None,
            'mean_output_tokens':sum(t['output_tokens'] for t in good)/len(good) if good else None,
            'mean_wall_seconds':sum(t['wall_seconds'] for t in good)/len(good) if good else None,
            'checker_abstentions':dict(Counter(r['checker']['reason'] for r in group if r['checker'] and r['prediction'] is None))}
    for arm in ['direct','reasoned','reasoned_long','thinking']:
        t=paired_test(result['arms'][arm]['pair_vector'],result['arms']['read']['pair_vector'],strata)
        t.update(a=arm,b='read_plus_rules',comparison='primary_equal_cap' if arm in ['direct','reasoned'] else 'registered_budget_sensitivity')
        result['registered_tests'].append(t)
    maximum=0.
    for rank,t in enumerate(sorted(result['registered_tests'],key=lambda t:t['p'])):
        maximum=max(maximum,min(1.,(4-rank)*t['p']));t['holm_p']=maximum
    return result,rows


def tables(out,d):
    lines=[r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Arm & Cap & Valid & False & Pair & Parsed & Cap hits \\',r'\midrule']
    costs=[r'\begin{tabular}{lrrr}',r'\toprule',r'Arm & Mean input & Mean output & Optimistic pair bound \\',r'\midrule']
    for arm,m in d['arms'].items():
        lines.append(' & '.join([LABELS[arm],str(ARMS[arm][0]),f"{m['valid_correct']}/128",f"{m['false_correct']}/128",f"{m['pair_correct']}/128",f"{m['covered']}/256",str(m['cap_hits'])])+r' \\')
        costs.append(' & '.join([LABELS[arm],f"{m['mean_input_tokens']:.1f}" if m['mean_input_tokens'] is not None else '---',f"{m['mean_output_tokens']:.1f}" if m['mean_output_tokens'] is not None else '---',f"{m['optimistic_pair_correct']}/128"])+r' \\')
    tests=[r'\begin{tabular}{lrrr}',r'\toprule',r'Read + rules minus & $\Delta$ (pp) & 95\% CI (pp) & $p_H$ \\',r'\midrule']
    for t in d['registered_tests']:
        tests.append(' & '.join([LABELS[t['a']],f"{100*t['delta']:+.2f}",f"[{100*t['ci95'][0]:.2f}, {100*t['ci95'][1]:.2f}]",r'$<.0001$' if t['holm_p']<.0001 else f"{t['holm_p']:.4f}"])+r' \\')
    for name,content in [('generated_table.tex',lines),('generated_costs.tex',costs),('generated_tests.tex',tests)]:
        (out/name).write_bytes(('\n'.join(content+[r'\bottomrule',r'\end{tabular}'])+'\n').encode())


def analyze(root,out):
    reg=json.loads((root/'registration.json').read_text());cases=json.loads((root/'cases.json').read_text())
    assert sha(root/'cases.json')==reg['cases_sha256'] and sha(root/'requests.jsonl')==reg['requests_sha256']
    for name,h in reg['code'].items(): assert sha(root/'executed_source'/name)==h
    for c in cases: assert sha(root/c['image'])==c['image_sha256']
    requests=read_jsonl(root/'requests.jsonl');calls=[];audits=[]
    for i in range(8):
        folder=root/f'worker_{i}';summary=json.loads((folder/'summary.json').read_text());identity=json.loads((folder/'identity.json').read_text());chunk=read_jsonl(folder/'calls.jsonl')
        assert summary['status']=='complete' and summary['calls']==160 and summary['calls_sha256']==sha(folder/'calls.jsonl')
        assert identity['registration_sha256']==sha(root/'registration.json') and identity['code']==reg['code']
        assert [c['key'] for c in chunk]==identity['requests']==[r['key'] for r in requests[i::8]]
        assert not (folder/'.running').exists() and not (folder/'pending.json').exists()
        calls.extend(chunk);audits.append({'worker':i,'calls_sha256':sha(folder/'calls.jsonl'),'errors':summary['errors']})
    result,rows=calculate(reg,cases,requests,calls)
    out.mkdir(parents=True,exist_ok=False)
    for name in ['cases.json','requests.jsonl']: shutil.copyfile(root/name,out/name)
    # Public arguments retain all numerical settings, not private filesystem paths.
    design=json.loads(json.dumps(reg));design['server_args'][0]='vllm';design['server_args'][2]=reg['model_id']
    design['original_registration_sha256']=sha(root/'registration.json')
    write(out/'study_design.json',design);write(out/'analysis.json',result)
    for name,records in [('calls.jsonl',calls),('predictions.jsonl',rows)]:
        (out/name).write_bytes((''.join(canonical(r)+'\n' for r in records)).encode())
    tables(out,result)
    write(out/'provenance.json',{'original_registration_sha256':sha(root/'registration.json'),'workers':audits,'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'p_display_threshold':.0001})
    print(json.dumps({a:{k:v for k,v in m.items() if k!='pair_vector'} for a,m in result['arms'].items()}))


def replay(out):
    prov=json.loads((out/'provenance.json').read_text())
    for name,h in prov['files'].items(): assert sha(out/name)==h,'file drift: '+name
    d,rows=calculate(json.loads((out/'study_design.json').read_text()),json.loads((out/'cases.json').read_text()),read_jsonl(out/'requests.jsonl'),read_jsonl(out/'calls.jsonl'))
    assert d==json.loads((out/'analysis.json').read_text()) and rows==read_jsonl(out/'predictions.jsonl')
    print('PASS: all 1280 larger-model calls, five arms, failure denominators, budget traces and four corrected contrasts replay.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--output',type=Path);p.add_argument('--replay',type=Path)
    a=p.parse_args()
    if a.replay: replay(a.replay)
    elif a.root and a.output: analyze(a.root,a.output)
    else: p.error('provide root/output or replay')
