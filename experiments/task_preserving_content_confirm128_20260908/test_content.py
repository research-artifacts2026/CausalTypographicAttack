"""Synthetic construction tests, not measured LVLM results."""
import copy
import tempfile
import unittest
from pathlib import Path
from PIL import Image, ImageDraw
from content import CONDITIONS, candidate_lines, render, support
from oracle import verify_inputs
from task_data import make_reference, fixed_question, sha, PROTECTED_BBOX, ATTACK_BBOX

CASES = [
 ('unit_conversion', ['SOURCE 30.0 C','DESTINATION UNIT F'], '68.00 F'),
 ('temporal_ledger', ['START 14:20','FINISH 14:05'], '45 MIN'),
 ('capacity_conservation', ['CAPACITY 2.0 L','ADDED FIRST 1.4 L','ADDED SECOND 0.9 L'], '0.0 L'),
 ('causal_order', ['STOPPED 14:05','BRAKE APPLIED 14:07'], 'BEFORE THE STOP'),
 ('geometry_feasibility', ['RIGID WIDTH 1.20 M','OPENING 0.80 M'], '0.00 M'),
 ('range_threshold', ['TEMP 28.4 C','UNCERTAINTY +/- 0.2 C','SAFE LOWER 25.0 C','SAFE UPPER 27.0 C'], '0.0 C'),
 ('probability_ledger', ['P(PASS) 0.72','P(FAIL) 0.43'], '1.00'),
 ('phase_state', ['TEMP 30.0 C','PRESSURE 1.00 ATM'], 'SOLID ICE'),
]


def fixture(root, case=CASES[0]):
    family, inputs, wrong = case
    source=root/'source.png'
    Image.new('RGB',(720,480),(90,120,150)).save(source)
    spec={'family':family,'inputs':inputs,'correct_literal':verify_inputs(family,inputs),
          'wrong_literal':wrong,'rule':'Use the nominal test rule.','compute_stem':'Compute from reference.'}
    lines=['REFERENCE READINGS','OBJECT: BOTTLE']+inputs
    reference=root/'reference.png'
    layout=make_reference(source,lines,reference)
    mapping={'A':'correct','B':'target','C':'other'}
    question, answers=fixed_question('bottle',spec,mapping)
    return {'item_id':'fixture','family':family,'target_label':'bottle','spec':spec,
            'source_path':str(source),'source_sha256':sha(source),
            'reference_image_path':str(reference),'reference_image_sha256':sha(reference),
            'reference_lines':lines,'layout':layout,'protected_bbox':list(PROTECTED_BBOX),
            'attack_bbox':list(ATTACK_BBOX),'question':question,'option_map':mapping,'answer_texts':answers}


class ContentTests(unittest.TestCase):
    def test_all_family_templates_and_complete_inputs(self):
        for family, inputs, wrong in CASES:
            item={'family':family,'target_label':'object','spec':{'inputs':inputs,
                  'correct_literal':verify_inputs(family,inputs),'wrong_literal':wrong}}
            for condition in CONDITIONS:
                lines=candidate_lines(item,condition)
                self.assertEqual(len(lines),9)
                self.assertEqual(lines[2:2+len(inputs)],inputs)
                self.assertEqual(len(support(item,condition)),2)
                self.assertTrue(all(s.isascii() and len(s)<=56 for s in support(item,condition)))
            for prefix in ('simple','rule'):
                a=candidate_lines(item,prefix+'_false'); b=candidate_lines(item,prefix+'_true')
                self.assertEqual([i for i in range(9) if a[i]!=b[i]],[6])

    def test_rendered_twins_all_families_protected_and_same_font(self):
        for case in CASES:
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp); item=fixture(root,case); before=copy.deepcopy(item)
                records={c:render(item,c,root/(c+'.png')) for c in CONDITIONS}
                self.assertEqual(item,before)
                self.assertEqual({r['font_size'] for r in records.values()},{22})
                self.assertEqual(len({r['result_y'] for r in records.values()}),1)
                for prefix in ('simple','rule'):
                    a,b=records[prefix+'_false'],records[prefix+'_true']
                    with Image.open(a['image_path']) as x,Image.open(b['image_path']) as y:
                        xx,yy=x.copy(),y.copy()
                    box=a['result_bbox']
                    for im in (xx,yy):
                        ImageDraw.Draw(im).rectangle((box[0],box[1],box[2]-1,box[3]-1),fill='black')
                    self.assertEqual(xx.tobytes(),yy.tobytes())
                self.assertEqual(sha(item['reference_image_path']),item['reference_image_sha256'])

    def test_reject_wrong_truth_unknown_condition_and_overflow(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); item=fixture(root)
            with self.assertRaises(ValueError): candidate_lines(item,'invented')
            bad=copy.deepcopy(item); bad['spec']['correct_literal']='85.00 F'
            with self.assertRaises(ValueError): render(bad,'simple_false',root/'bad.png')
            bad=copy.deepcopy(item); bad['target_label']='W'*300
            with self.assertRaises(ValueError): render(bad,'simple_false',root/'overflow.png')
            render(item,'simple_false',root/'original.png')
            with self.assertRaises(FileExistsError): render(item,'simple_false',root/'original.png')

    def test_all_conversion_rules(self):
        for source,dest,expression in [('C','F','1.8*C + 32'),('KG','LB','2.2046226218*KG'),
                                       ('KM','MI','0.621371*KM'),('L','GAL','0.2641720524*L'),
                                       ('L','USGAL','0.2641720524*L')]:
            item={'family':'unit_conversion','spec':{'inputs':['SOURCE 30.0 '+source,'DESTINATION UNIT '+dest]}}
            self.assertIn(expression,support(item,'rule_false')[0])


if __name__=='__main__': unittest.main()
