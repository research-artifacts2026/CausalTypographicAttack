"""Run fixed requests against this experiment's local GPU service; no paid APIs."""
import argparse, base64, concurrent.futures, json, os, time
from pathlib import Path
import sys
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cta.verification_workbench import now, sha, read_jsonl, append
from scripts.strong_model_lib import code_hashes, payload, write

URL='http://127.0.0.1:18127/v1/chat/completions'


def worker(root, index, requests, reg):
    folder=root/f'worker_{index}';folder.mkdir(exist_ok=False)
    (folder/'.running').write_text('active')
    write(folder/'identity.json',{'registration_sha256':sha(root/'registration.json'),'code':code_hashes(),'requests':[r['key'] for r in requests]})
    calls=[]
    try:
        for r in requests:
            picture=(root/r['image']).resolve()
            assert picture.is_relative_to(root.resolve()) and sha(picture)==r['image_sha256']
            image='data:image/png;base64,'+base64.b64encode(picture.read_bytes()).decode()
            body=json.dumps(payload(r,image)).encode()
            started=now();clock=time.perf_counter()
            write(folder/'pending.json',{'key':r['key'],'started_at':started})
            response=None;error=None
            try:
                request=urllib.request.Request(URL,data=body,headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=900) as result: response=json.load(result)
                assert response['model']=='qwen35-27b-cta' and len(response['choices'])==1
            except Exception as exc:
                error=type(exc).__name__+': '+str(exc)
            raw='';reasoning='';trace={}
            if not error:
                choice=response['choices'][0];message=choice['message'];usage=response['usage']
                raw=message.get('content') or ''
                reasoning=message.get('reasoning') or message.get('reasoning_content') or ''
                trace={'input_tokens':usage['prompt_tokens'],'output_tokens':usage['completion_tokens'],
                       'finish_reason':choice['finish_reason'],'cap_hit':choice['finish_reason']=='length',
                       'wall_seconds':time.perf_counter()-clock,'usage':usage}
            call={'key':r['key'],'request':r,'raw':raw,'reasoning':reasoning,'response':response,
                  'error':error,'trace':trace,'started_at':started,'finished_at':now()}
            append(folder/'calls.jsonl',call);calls.append(call)
            (folder/'pending.json').unlink()
            print(f'worker {index}: {len(calls)}/{len(requests)}, errors {sum(bool(c["error"]) for c in calls)}',flush=True)
        write(folder/'summary.json',{'status':'complete','calls':len(calls),'errors':sum(bool(c['error']) for c in calls),'calls_sha256':sha(folder/'calls.jsonl'),'finished_at':now()})
    finally:
        (folder/'.running').unlink(missing_ok=True)


def run(root):
    reg=json.loads((root/'registration.json').read_text())
    assert code_hashes()==reg['code'] and sha(root/'requests.jsonl')==reg['requests_sha256']
    requests=read_jsonl(root/'requests.jsonl')
    assert len(requests)==1280 and all(not (root/f'worker_{i}').exists() for i in range(8))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(worker,root,i,requests[i::8],reg) for i in range(8)]
        for f in futures: f.result()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)
