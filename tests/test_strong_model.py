import copy
import unittest
from tests.test_symbolic_confirmation import ConfirmationTests
from scripts.strong_model_lib import requests_for, payload
from scripts.analyze_strong_model import calculate


class StrongModelTests(unittest.TestCase):
    def fixture(self):
        _,cases,_,_=ConfirmationTests().fixture()
        reg={'frozen_at':'2026-01-01','model_id':'Qwen/Qwen3.5-27B'}
        requests=requests_for(cases);ci={(c['item_id'],c['state']):c for c in cases};calls=[]
        for r in requests:
            c=ci[r['item_id'],r['state']];letter='A' if c['gold']=='consistent' else 'B'
            raw=c['fields'] if r['arm']=='read' else letter if r['arm']=='direct' else 'Calculation\nFINAL: '+letter
            response={'model':'qwen35-27b-cta','choices':[{'message':{'content':raw,'reasoning':''},'finish_reason':'stop'}],
                      'usage':{'prompt_tokens':30,'completion_tokens':10}}
            calls.append({'key':r['key'],'request':r,'raw':raw,'reasoning':'','response':response,'error':None,
                          'started_at':'2026-01-02','trace':{'input_tokens':30,'output_tokens':10,'finish_reason':'stop','cap_hit':False,'wall_seconds':.1}})
        return reg,cases,requests,calls

    def test_complete_and_missing_final_bound(self):
        reg,cases,requests,calls=self.fixture()
        call=next(c for c in calls if c['request']['arm']=='thinking')
        call['raw']='unfinished';call['response']['choices'][0]['message']['content']='unfinished'
        d,rows=calculate(reg,cases,requests,calls)
        self.assertEqual(d['actual_calls'],1280)
        self.assertEqual(d['arms']['thinking']['pair_correct'],127)
        self.assertEqual(d['arms']['thinking']['optimistic_pair_correct'],128)
        self.assertEqual(d['arms']['thinking']['n'],256)
        self.assertEqual(len(d['registered_tests']),4)

    def test_rejects_missing_calls_and_budget_drift(self):
        reg,cases,requests,calls=self.fixture()
        with self.assertRaisesRegex(ValueError,'incomplete'): calculate(reg,cases,requests,calls[:-1])
        altered=copy.deepcopy(requests);altered[0]['tokens']=1
        with self.assertRaisesRegex(ValueError,'budget'): calculate(reg,cases,altered,calls)

    def test_runtime_error_not_optimistically_repaired(self):
        reg,cases,requests,calls=self.fixture()
        call=next(c for c in calls if c['request']['arm']=='thinking');call['error']='synthetic outage';call['response']=None;call['trace']={}
        d,_=calculate(reg,cases,requests,calls)
        self.assertEqual(d['arms']['thinking']['optimistic_pair_correct'],127)
        self.assertEqual(d['arms']['thinking']['runtime_errors'],1)

    def test_retains_thinking_and_caps_in_request(self):
        _,cases,_,_=ConfirmationTests().fixture()
        requests=requests_for(cases)
        thinking=next(r for r in requests if r['arm']=='thinking')
        direct=next(r for r in requests if r['arm']=='direct')
        self.assertTrue(payload(thinking,'data:image/png;base64,AA')['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(thinking['tokens'],4096)
        self.assertFalse(payload(direct,'data:image/png;base64,AA')['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(direct['tokens'],384)


if __name__=='__main__': unittest.main()
