"""Execute an immutable request shard once, retaining all failures."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.channel_study_lib import code_hashes,infer_optional
from cta.verification_workbench import sha,digest,now,write_json,read_jsonl,append


def run(root,model_name,shard):
    reg=json.loads((root/'registration.json').read_text())
    if code_hashes()!=reg['code']: raise ValueError('registered execution code changed')
    manifest=root/f'requests_{shard}.jsonl'
    if sha(manifest)!=reg['request_hashes'][str(shard)]: raise ValueError('request manifest changed')
    requests=read_jsonl(manifest)
    cfg=reg['models'][model_name]['config']
    folder=root/f'{model_name}_{shard}';folder.mkdir(exist_ok=True)
    lock=folder/'.running';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    try:
        ident={'registration':sha(root/'registration.json'),'requests':sha(manifest),
               'code':code_hashes(),'config':digest(cfg),'python':sys.version}
        identity=folder/'identity.json'
        if identity.exists() and json.loads(identity.read_text())!=ident: raise ValueError('resume identity mismatch')
        if not identity.exists(): write_json(identity,ident)
        if (folder/'pending.json').exists(): raise ValueError('unknown interrupted call; no silent retry')
        existing=read_jsonl(folder/'calls.jsonl');cache={c['key']:c for c in existing}
        if len(cache)!=len(existing):raise ValueError('duplicate calls')
        if set(cache)-{r['key'] for r in requests}:raise ValueError('unexpected calls')
        if (folder/'summary.json').exists():
            done=json.loads((folder/'summary.json').read_text())
            if done['calls_sha256']!=sha(folder/'calls.jsonl'):raise ValueError('finished journal changed')
        model=None
        for req in requests:
            if req['image']:
                image=root/req['image']
                if not image.resolve().is_relative_to(root.resolve()) or sha(image)!=req['image_sha256']:raise ValueError('image changed')
            else:image=None
            if req['key'] in cache:
                if cache[req['key']]['request']!=req:raise ValueError('request drift')
                continue
            if model is None:
                from cta.model import build_model_adapter
                model=build_model_adapter(cfg)
                write_json(folder/'model.json',model.provenance())
            started=now();write_json(folder/'pending.json',{'key':req['key'],'started_at':started})
            try:
                raw,trace=infer_optional(model,str(image) if image else None,req['prompt'],req['tokens'])
                error=None
            except Exception as exc:
                raw='';error=f'{type(exc).__name__}: {exc}';trace={}
            call={'key':req['key'],'request':req,'raw':raw,'error':error,
                  'trace':trace,'started_at':started,'finished_at':now()}
            append(folder/'calls.jsonl',call);cache[req['key']]=call
            (folder/'pending.json').unlink()
            print(f'{len(cache)}/{len(requests)} calls retained',flush=True)
        write_json(folder/'summary.json',{'status':'complete','calls':len(cache),
            'errors':sum(bool(c['error']) for c in cache.values()),
            'calls_sha256':sha(folder/'calls.jsonl'),'finished_at':now()})
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--model',choices=['qwen7','qwen3'],required=True);p.add_argument('--shard',type=int,choices=[0,1],required=True)
    a=p.parse_args();run(a.root,a.model,a.shard)
