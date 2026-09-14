"""Create an outcome-blind review packet; keep the answer key outside the ZIP."""
import argparse,csv,hashlib,html,io,json,shutil,zipfile
from collections import defaultdict
from pathlib import Path


def h(value):return hashlib.sha256(value.encode()).hexdigest()


def build(root,output):
    cases=json.loads((root/'cases.json').read_text());buckets=defaultdict(set)
    for c in cases:
        bucket=(c['study'],c['source'] if c['study']=='media' else '',c['family'])
        buckets[bucket].add(c['item_id'])
    selected=set()
    for bucket,ids in sorted(buckets.items()):
        selected.update((bucket[0],i) for i in sorted(ids,key=lambda i:h('blind-review-v1:'+i))[:2])
    rows=[];key=[]
    output.mkdir(parents=True,exist_ok=False);(output/'images').mkdir()
    requests={r['key']:r for n in [0,1] for r in map(json.loads,(root/f'requests_{n}.jsonl').read_text().splitlines())}
    for c in cases:
        if (c['study'],c['item_id']) not in selected:continue
        rid='R-'+h('blind-review-case-v1:'+c['study']+c['item_id']+c['state'])[:12]
        req=next(r for r in requests.values() if r['study']==c['study'] and r['item_id']==c['item_id'] and r['state']==c['state'] and r['arm']==('scene' if c['study']=='media' else 'none'))
        dest='images/'+rid+'.png';shutil.copyfile(root/c['image'],output/dest)
        rows.append({'review_id':rid,'image':dest,'prompt':req['prompt'],'assumptions':c['assumption'],
                     'target_category':c.get('target_label'),'task':'Assess the displayed record' if c['study']=='media' else 'Assess the record beneath the requested object'})
        key.append({'review_id':rid,'study':c['study'],'item_id':c['item_id'],'state':c['state'],'gold':c['gold'],'family':c['family'],'source':c['source']})
    rows.sort(key=lambda r:r['review_id']);key.sort(key=lambda r:r['review_id'])
    if len(rows)!=96:raise ValueError('expected 96 blinded cases')
    (output/'cases.json').write_bytes((json.dumps(rows,indent=2)+'\n').encode())
    answer_key=output.with_name(output.name+'_PRIVATE_KEY.json')
    answer_key.write_bytes((json.dumps(key,indent=2)+'\n').encode())
    columns=['reviewer_id','review_id','judgment','assumptions_sufficient','readability','target_visible','notes']
    with (output/'annotations_blank.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        writer.writerows({'review_id':r['review_id']} for r in rows)
    template='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Independent record review</title>
<style>body{font:16px system-ui;max-width:1250px;margin:24px auto;padding:0 16px;color:#172331}header{position:sticky;top:0;background:white;padding:12px 0;border-bottom:1px solid #aaa}button,input,select,textarea{font:inherit;padding:7px;margin:5px}#picture{max-width:100%;max-height:70vh;border:1px solid #999}label{display:block}textarea{width:90%;height:75px}.notice{background:#eef3f8;padding:14px}#progress{font-weight:bold}</style>
<header><b>Independent record review</b> — <span id="progress"></span><br><label>Reviewer ID (use a pseudonym) <input id="reviewer" placeholder="Reviewer-1"></label><button id="prev">Previous</button><select id="caseSelect"></select><button id="next">Next</button><button id="download">Download my CSV</button></header>
<p class="notice">Judge independently, without model answers or discussion with another reviewer. These are hypothetical records, not measurements recovered from photographs. Evaluate under the stated assumptions; flag assumptions that do not suffice. Do not infer that an unusual image implies an invalid record. For two-panel cases, assess only the record under the requested object. Leave uncertain cases as ambiguous. No field is pre-annotated.</p>
<h2 id="caseTitle"></h2><p id="task"></p><p id="prompt"></p><img id="picture" alt="Record to review">
<label>Record judgment <select id="judgment"><option value="">Unreviewed</option><option>consistent</option><option>inconsistent</option><option>ambiguous</option><option>unreadable</option><option>unsupported</option></select></label>
<label>Are the supplied assumptions sufficient? <select id="assumptions_sufficient"><option value="">Unreviewed</option><option>yes</option><option>no</option><option>unclear</option></select></label>
<label>Are all relevant record fields readable? <select id="readability"><option value="">Unreviewed</option><option>yes</option><option>no</option><option>unclear</option></select></label>
<label>Can you identify the requested object? <select id="target_visible"><option value="">Unreviewed</option><option>yes</option><option>no</option><option>unclear</option><option>not_applicable</option></select></label>
<label>Explanation / missing assumption / ambiguous field <textarea id="notes"></textarea></label>
<p>Answers are saved only in this browser on this computer. Download the CSV before closing or moving the folder. This page sends no data to a server.</p>
<script>const cases=__CASES__;const fields=['judgment','assumptions_sufficient','readability','target_visible','notes'];const $=id=>document.getElementById(id);let index=0;let saved={};try{saved=JSON.parse(localStorage.getItem('record-review-v1')||'{}')}catch(e){};$('reviewer').value=saved.reviewer||'';
function persist(){saved.reviewer=$('reviewer').value;saved[cases[index].review_id]=Object.fromEntries(fields.map(f=>[f,$(f).value]));try{localStorage.setItem('record-review-v1',JSON.stringify(saved))}catch(e){};status()}
function status(){const n=cases.filter(c=>saved[c.review_id]?.judgment).length;$('progress').textContent=`${n} of ${cases.length} reviewed`}
function show(){const c=cases[index];$('caseSelect').value=index;$('caseTitle').textContent=c.review_id;$('task').textContent=c.task;$('prompt').textContent=c.prompt;$('picture').src=c.image;for(const f of fields)$(f).value=saved[c.review_id]?.[f]||'';status()}
cases.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${i+1}. ${c.review_id}`;$('caseSelect').appendChild(o)});fields.forEach(f=>$(f).addEventListener('input',persist));$('reviewer').oninput=persist;$('prev').onclick=()=>{persist();index=Math.max(0,index-1);show()};$('next').onclick=()=>{persist();index=Math.min(cases.length-1,index+1);show()};$('caseSelect').onchange=()=>{persist();index=Number($('caseSelect').value);show()};
$('download').onclick=()=>{persist();const columns=['reviewer_id','review_id',...fields];const quote=v=>'"'+String(v||'').replaceAll('"','""')+'"';const lines=[columns,...cases.map(c=>[saved.reviewer,c.review_id,...fields.map(f=>saved[c.review_id]?.[f]||'')])];const blob=new Blob(['\\ufeff'+lines.map(r=>r.map(quote).join(',')).join('\\r\\n')],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='independent_annotations.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};show();</script></html>'''
    (output/'review.html').write_text(template.replace('__CASES__',json.dumps(rows).replace('<','\\u003c')),encoding='utf-8')
    (output/'README.txt').write_text('Unzip this folder and open review.html. Each reviewer should use a separate copy/browser profile and choose a different pseudonym. Complete all 96 cases independently and download the CSV. No model outputs or reference labels are included. Selection: two stable hash-ordered media items per source/family and two object-swap items per family, retaining both states, then hash-shuffled case order. This review covers a sample, not the full benchmark. Uncertain or conflicting judgments require adjudication; do not replace missing ratings with model output.\n',encoding='utf-8')
    with zipfile.ZipFile(output.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(output).as_posix())
    print(json.dumps({'cases':len(rows),'zip':str(output.with_suffix('.zip')),'private_key_outside_packet':str(answer_key)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.root,a.output)
