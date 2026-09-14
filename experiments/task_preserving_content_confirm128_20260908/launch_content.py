"""Scoped two-worker content scheduler; never reclaims another process's GPU."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from run_content import verify_registration
from search import write_or_verify, file_hash


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--base',type=Path,required=True)
    a=p.parse_args(); root,base=a.root.resolve(),a.base.resolve(); code=Path(__file__).resolve().parent
    verify_registration(base)
    preflight=json.loads((base/'preflight.json').read_text())
    if (preflight['status']!='pass' or preflight['victim_calls_before_gate']!=0 or
            preflight['registration_sha256']!=file_hash(base/'registration.json')):
        raise ValueError('Independent pre-inference integrity gate missing or stale')
    if (base/'launch.json').exists(): raise FileExistsError('Do not start another scheduler')
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    memory={int(x.split(',')[0]):int(x.split(',')[1]) for x in usage.splitlines()}
    if memory[6]>500 or memory[7]>500: raise RuntimeError('GPUs6/7 are occupied; existing jobs will not be interrupted')
    workers=[]
    for model,gpu in [('qwen7',6),('qwen3vl8',7)]:
        env=os.environ.copy(); env.pop('PYTHONPATH',None)
        env.update(CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
        if model=='qwen3vl8': env['PYTHONPATH']='/disk2/fangxinyue/cta_crossvl_env/lib/python3.10/site-packages'
        cmd=['/disk2/fangxinyue/.venv/bin/python',str(code/'run_content.py'),'run','--root',str(root),'--base',str(base),'--model',model]
        with (base/(model+'_console.log')).open('ab') as log:
            process=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append((model,gpu,cmd,process))
    write_or_verify(base/'launch.json',{'launcher_sha256':file_hash(__file__),'jobs':[
        {'model':m,'gpu':g,'command':c,'pid':proc.pid} for m,g,c,proc in workers]})
    while True:
        state={'status':'running','jobs':[{'model':m,'pid':pr.pid,'exit_code':pr.poll()} for m,g,c,pr in workers]}
        if all(pr.poll() is not None for m,g,c,pr in workers):
            complete=all(pr.returncode==0 and (base/m/'complete.json').exists() and
                         json.loads((base/m/'complete.json').read_text())['status']=='complete' for m,g,c,pr in workers)
            state['status']='complete' if complete else 'attention_required'
        pending=base/'scheduler_status.pending.json'
        pending.write_text(json.dumps(state,indent=2)); pending.replace(base/'scheduler_status.json')
        if state['status']!='running': break
        time.sleep(3)
    if state['status']!='complete': raise RuntimeError('Content collection incomplete; inspect retained logs')


if __name__=='__main__': main()
