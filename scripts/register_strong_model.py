"""Register all arms, assets, checkpoint hashes and scoring before inference."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cta.verification_workbench import sha, now, canonical
from scripts.strong_model_lib import ARMS, MODEL, code_hashes, requests_for, write


def register(source, checkpoint, service, output):
    old=json.loads((source/'registration.json').read_text())
    assert sha(source/'cases.json')==old['cases_sha256']
    cases=json.loads((source/'cases.json').read_text())
    assert len(cases)==256 and len({c['item_id'] for c in cases})==128
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(source/'cases.json',output/'cases.json')
    for c in cases:
        original=source/c['image']; assert sha(original)==c['image_sha256']
        target=output/c['image'];target.parent.mkdir(exist_ok=True)
        if not target.exists(): shutil.copyfile(original,target)
    requests=requests_for(cases)
    assert len(requests)==1280
    (output/'requests.jsonl').write_bytes((''.join(canonical(r)+'\n' for r in requests)).encode())
    index=json.loads((checkpoint/'model.safetensors.index.json').read_text())
    weights=set(index['weight_map'].values())
    assert weights and all((checkpoint/n).is_file() for n in weights)
    paths=[f for f in checkpoint.iterdir() if f.is_file() and
           (f.suffix in ['.safetensors','.json','.jinja'] or f.name in ['merges.txt','tokenizer.model'])]
    checkpoint_hashes={f.name:sha(f) for f in sorted(paths)}
    code=code_hashes(); repo=Path(__file__).resolve().parents[1]
    for name in code:
        target=output/'executed_source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(repo/name,target)
    launch=json.loads((service/'launch.json').read_text())
    versions={}
    for package in ['vllm','transformers','torch','numpy']:
        versions[package]=importlib.metadata.version(package)
    write(output/'registration.json',{
        'frozen_at':now(),'model_id':MODEL,'checkpoint_files_sha256':checkpoint_hashes,
        'parent_registration_sha256':sha(source/'registration.json'),'cases_sha256':sha(output/'cases.json'),
        'requests_sha256':sha(output/'requests.jsonl'),'code':code,'versions':versions,
        'server_args':launch['args'],'server_gpus':launch['gpus'],'service_launch_sha256':sha(service/'launch.json'),
        'items':128,'cases':256,'calls':1280,'workers':8,'arms':ARMS,
        'selection':'All 128 items and both states of the prior completed symbolic confirmation, with no outcome-based filtering. New-model inference is prospective; scenes have already been evaluated on smaller models.',
        'primary':'Pair accuracy, requiring valid and false states both correct. Read-plus-fixed-rules versus direct and reasoned under 384-token caps.',
        'secondary_registered':'Read-plus-rules versus reasoned at 2048 tokens and thinking at 4096 tokens. Unequal caps and thinking mode are explicit sensitivity arms, not matched-compute comparisons.',
        'tests':'Exact two-sided McNemar; Holm across all four registered contrasts. 10000 source/family-stratified paired bootstrap replicates, seed 20260914; marginal 95% intervals.',
        'format_sensitivity':'Additionally report an optimistic bound crediting every unparsed non-runtime-error neural output as correct. No parser relaxation or primary rescoring.',
        'stopping':'All 1280 registered calls once; runtime and parsing failures count incorrect; cap hits are retained and reported without selectively extending them. No automatic retries or significance stopping. Infrastructure interruptions require a retained incident record.',
        'budget':'Direct, reasoned and read:384 output tokens; longer reasoned:2048; thinking:4096 including reasoning tokens. Actual token usage retained. Greedy controlled comparison, not vendor-recommended sampled benchmark settings.',
        'scope':'Single larger open checkpoint; same known schemas and previously evaluated scenes. Does not establish closed-model, new-schema or human-validated generalization.'})
    print(json.dumps({'registration_sha256':sha(output/'registration.json'),'calls':1280,'items':128}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['source','checkpoint','service','output']: p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();register(a.source,a.checkpoint,a.service,a.output)
