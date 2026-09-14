"""Fixed design for the Qwen3.5-27B cross-model replication."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cta.verification_workbench import canonical, sha, source_code_hashes
from scripts.symbolic_confirmation_lib import prompts

MODEL = 'Qwen/Qwen3.5-27B'
SERVED_MODEL = 'qwen35-27b-cta'
ARMS = {'direct': (384, False), 'reasoned': (384, False), 'read': (384, False),
        'reasoned_long': (2048, False), 'thinking': (4096, True)}
LABELS = {'direct': 'Direct', 'reasoned': 'Reasoned', 'read': 'Read + rules',
          'reasoned_long': 'Reasoned, longer', 'thinking': 'Thinking'}


def write(path, obj):
    path.write_bytes((json.dumps(obj, indent=2, ensure_ascii=False) + '\n').encode())


def code_hashes():
    root=Path(__file__).resolve().parents[1]
    code=source_code_hashes()
    for name in ['strong_model_lib.py','register_strong_model.py','run_strong_model.py',
                 'analyze_strong_model.py','symbolic_confirmation_lib.py','analyze_verification_diagnostic.py']:
        code['scripts/'+name]=sha(root/'scripts'/name)
    return code


def requests_for(cases):
    requests=[]
    ids=sorted({c['item_id'] for c in cases})
    case_map={(c['item_id'],c['state']):c for c in cases}
    names=list(ARMS)
    for j,item in enumerate(ids):
        arms=names[j%5:]+names[:j%5]
        for state in ['record_true','record_false']:
            c=case_map[item,state]
            ps=prompts(c['assumption'],c['option_map'])
            for arm in arms:
                tokens,thinking=ARMS[arm]
                requests.append({'key':canonical([item,state,arm]),'item_id':item,'state':state,'arm':arm,
                    'image':c['image'],'image_sha256':c['image_sha256'],
                    'prompt':ps[arm if arm in ps else 'reasoned'], 'tokens':tokens,
                    'enable_thinking':thinking,'temperature':0.0,'seed':20260914})
    return requests


def payload(request, image_data):
    return {'model':SERVED_MODEL,'messages':[{'role':'user','content':[
        {'type':'image_url','image_url':{'url':image_data}}, {'type':'text','text':request['prompt']}]}],
        'max_tokens':request['tokens'],'temperature':request['temperature'],'seed':request['seed'],
        'top_p':1.0,'top_k':-1,'presence_penalty':0.0,'repetition_penalty':1.0,
        'chat_template_kwargs':{'enable_thinking':request['enable_thinking']},'stream':False}
