import pytest
from scripts.launch_verification_gradio import outcome, answers_table, metrics_table, shared_probe_table
from cta.verification_workbench import score, CONDITIONS, STRATEGIES, canonical


def summary(answers):
    return score([{'item_id':'x'}], [{'item_id':'x','condition':c,'strategy':'direct','parsed':a,'raw':'A'}
                                    for c,a in zip(CONDITIONS,answers)], ['direct'])


def test_invalid_zero_accuracy_can_mean_attack_success():
    s = summary(['absent','consistent','consistent'])
    row = s['strategies']['direct']
    assert row['accuracy']['record_false']['k'] == 0
    assert outcome(row).startswith('攻击成功')
    assert len(metrics_table(s)[0]) == 10


def test_failed_control_is_not_a_defense_success():
    row = summary(['consistent','consistent','inconsistent'])['strategies']['direct']
    assert outcome(row).startswith('对照未通过')


def test_resisted_attack_and_ambiguous_answer_are_distinct():
    row = summary(['absent','consistent','inconsistent'])['strategies']['direct']
    assert outcome(row).startswith('攻击未成功')
    row = summary(['absent','consistent',None])['strategies']['direct']
    assert outcome(row).startswith('无法判定')


def test_raw_answer_is_displayed_with_its_actual_option_map():
    rows=[{'item_id':'x','condition':'record_false','option_map':{'A':'consistent','B':'inconsistent','C':'absent'}}]
    predictions=[{'item_id':'x','condition':'record_false','strategy':'direct','raw':'A','parsed':'consistent'}]
    shown=answers_table(rows,predictions)[0]
    assert shown[2:] == ['A','记录有效','B · 记录无效','错误']


def test_shared_success_is_displayed_once_not_as_three_independent_verifications():
    predictions = [{'item_id':'x','condition':c,'strategy':s,'parsed':a,'raw':'A',
                    'read_match':True,'knowledge_correct':True}
                   for s in STRATEGIES for c,a in zip(CONDITIONS,['absent','consistent','consistent'])]
    calls = [{'key':canonical(['x',probe]),'request':{'prompt':'retained prompt'},'raw':'retained response','error':None}
             for probe in ['independent_read','independent_know']]
    shown = shared_probe_table(predictions,calls)
    assert len(shown) == 2
    assert [r[2] for r in shown] == ['PASS','PASS']
    scored = score([{'item_id':'x'}],predictions,list(STRATEGIES))
    for row in metrics_table(scored):
        assert row[2:9] == ['PASS','PASS','FAIL','FAIL','PASS','YES','YES']
        assert '%' not in ' '.join(map(str,row))
    assert scored['strategies']['direct']['eor']['rate'] == 1  # Original aggregate stays intact.
    predictions[-1]['knowledge_correct'] = False
    with pytest.raises(ValueError,match='inconsistent shared'):
        shared_probe_table(predictions,calls)


def test_ineligible_eor_is_not_displayed_as_successful_verification():
    shown = metrics_table(summary(['consistent','consistent','inconsistent']))[0]
    assert shown[6:9] == ['FAIL','NO','N/A']


def test_missing_or_failed_probe_is_never_a_displayed_pass():
    predictions=[{'item_id':'x','condition':'record_false','strategy':'direct',
                  'read_match':True,'knowledge_correct':True}]
    calls=[{'key':canonical(['x','independent_read']),'raw':'','request':{'prompt':'read'},'error':'timeout'}]
    shown=shared_probe_table(predictions,calls)
    assert shown[0][2].startswith('ERROR')
    assert shown[1][2].startswith('UNAVAILABLE')
