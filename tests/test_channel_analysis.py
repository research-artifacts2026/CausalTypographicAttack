import copy
import pytest
from cta.verification_workbench import canonical
from scripts.channel_study_lib import MAP,MEDIA_ARMS,BINDING_ARMS
from scripts.analyze_channel_study import calculate


@pytest.fixture(scope='module')
def example():
    reg={'frozen_at':'2026-01-01','media_primary':[{'a':'scene','b':'oracle_scene'},{'a':'oracle_scene','b':'oracle_text'},{'a':'scene','b':'record_only'}],
         'binding_primary':[{'a':'fields','b':'both'},{'a':'location','b':'both'}]}
    cases=[];requests=[];calls=[]
    for study,n,states,arms in [('media',128,['record_true','record_false'],MEDIA_ARMS+['read']),('binding',64,['target_left','target_right'],BINDING_ARMS+['localize','text_reasoning'])]:
        for i in range(n):
            item=f'{study}-{i:03d}'
            for s,state in enumerate(states):
                gold='consistent' if s==0 else 'inconsistent'
                cases.append({'study':study,'item_id':item,'state':state,'source':'coco' if i%2 else 'voc','family':str(i%8)})
                for arm in arms:
                    image=None if arm in ['oracle_text','text_reasoning'] else 'image.png'
                    target='VALUE 1'
                    r={'key':canonical([study,item,state,arm]),'study':study,'item_id':item,'state':state,'arm':arm,'image':image,'gold':('L' if s==0 else 'R') if arm=='localize' else gold,'option_map':MAP,'read_target':target}
                    raw=target if arm=='read' else (r['gold'] if arm=='localize' else ('A' if s==0 else 'B'))
                    # One false-record failure remains in the full denominator.
                    if study=='media' and i==0 and s==1 and arm=='scene':raw='A'
                    requests.append(r)
                    calls.append({'key':r['key'],'request':copy.deepcopy(r),'raw':raw,'error':None,'started_at':'2026-01-02','trace':{'image_supplied':False,'input_keys':['attention_mask','input_ids']} if image is None else {'image_supplied':True}})
    return reg,cases,requests,{'qwen7':calls,'qwen3':copy.deepcopy(calls)}


def test_full_denominators_and_separate_holm_families(example):
    result=calculate(*example)
    assert result['actual_calls']==4608
    for d in result['results'].values():
        m=d['media']['metrics']['scene']
        assert (m['pair_correct'],m['pair_n'],m['correct'],m['n'])==(127,128,255,256)
        assert d['media']['control_conditioned']['pair_failures']==1
        assert d['binding']['metrics']['none']['pair_correct']==64
    assert len([t for t in result['primary_tests'] if t['study']=='media'])==6
    assert len([t for t in result['primary_tests'] if t['study']=='binding'])==4


def test_duplicate_journal_entry_is_not_silently_dropped(example):
    reg,cases,requests,calls=example
    changed={**calls,'qwen7':calls['qwen7']+[calls['qwen7'][0]]}
    with pytest.raises(ValueError,match='call coverage'):calculate(reg,cases,requests,changed)


def test_image_free_claim_requires_no_visual_tensor(example):
    reg,cases,requests,calls=copy.deepcopy(example)
    call=next(c for c in calls['qwen7'] if c['request']['image'] is None)
    call['trace']['input_keys'].append('pixel_values')
    with pytest.raises(ValueError,match='visual tensor'):calculate(reg,cases,requests,calls)


def test_frozen_prompt_drift_aborts_scoring(example):
    reg,cases,requests,calls=copy.deepcopy(example)
    calls['qwen7'][0]['request']['prompt']='modified after freeze'
    with pytest.raises(ValueError,match='frozen request'):calculate(reg,cases,requests,calls)
