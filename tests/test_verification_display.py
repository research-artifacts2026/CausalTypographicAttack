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
    assert outcome(row).startswith('Incorrect: accepted')
    assert len(metrics_table(s)[0]) == 10


def test_failed_control_is_not_a_defense_success():
    row = summary(['consistent','consistent','inconsistent'])['strategies']['direct']
    assert outcome(row).startswith('Controls failed')


def test_resisted_attack_and_ambiguous_answer_are_distinct():
    row = summary(['absent','consistent','inconsistent'])['strategies']['direct']
    assert outcome(row).startswith('Correct: rejected')
    row = summary(['absent','consistent',None])['strategies']['direct']
    assert outcome(row).startswith('Undetermined')


def test_raw_answer_is_displayed_with_its_actual_option_map():
    rows=[{'item_id':'x','condition':'record_false','option_map':{'A':'consistent','B':'inconsistent','C':'absent'}}]
    predictions=[{'item_id':'x','condition':'record_false','strategy':'direct','raw':'A','parsed':'consistent'}]
    shown=answers_table(rows,predictions)[0]
    assert shown[2:] == ['A','Internally consistent','B · Internally inconsistent','Incorrect']


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


@pytest.fixture
def completed(tmp_path):
    from PIL import Image
    from cta.verification_demo import demo_rows
    from cta.verification_workbench import freeze_packet, load_packet, evaluate_packet
    image = tmp_path/'source.png'
    Image.new('RGB',(640,480),'gray').save(image)
    rows = demo_rows(image,'person','range_threshold',tmp_path/'render')
    packet = freeze_packet(rows,tmp_path/'packet',origin='display test')
    info, rows = load_packet(packet)
    class Model:
        def infer(self,image,prompt,max_new_tokens): return 'A'
    run = tmp_path/'evaluation'
    evaluate_packet(packet,run,{},lambda cfg: Model())
    return info, rows, run


def test_reference_uses_inclusive_containment_not_overlap():
    from scripts.launch_verification_gradio import reference_reason
    row = {'condition':'record_false','record':{'family':'range_threshold','parameters':{
        'false_temperature_c':26.0,'true_temperature_c':29.4,'uncertainty_c':0.4,'lower_c':26.4,'upper_c':32.4}}}
    assert '[25.6, 26.4]' in reference_reason(row)
    assert 'NOT fully inside' in reference_reason(row)
    row['record']['parameters']['false_temperature_c'] = 27.0
    with pytest.raises(ValueError, match='contradict'):
        reference_reason(row)


def test_replay_checks_actual_answers_even_if_the_score_is_unchanged(completed):
    import json
    from scripts.launch_verification_gradio import validate_saved_run
    from cta.verification_workbench import read_jsonl
    info, rows, run = completed
    validate_saved_run(rows,info,run)
    predictions = read_jsonl(run/'predictions.jsonl')
    predictions[0]['raw'] = 'B'  # Changing only the displayed raw answer used to evade score validation.
    (run/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in predictions))
    with pytest.raises(ValueError, match='actual call'):
        validate_saved_run(rows,info,run)


def test_replay_checks_journal_and_probe_flags(completed):
    import json
    from scripts.launch_verification_gradio import validate_saved_run
    from cta.verification_workbench import read_jsonl
    info, rows, run = completed
    predictions = read_jsonl(run/'predictions.jsonl')
    false = next(p for p in predictions if p['condition']=='record_false')
    false['read_match'] = not false['read_match']
    (run/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in predictions))
    with pytest.raises(ValueError, match='independent probes'):
        validate_saved_run(rows,info,run)
    with (run/'calls.jsonl').open('a') as file:
        file.write('\n')
    with pytest.raises(ValueError, match='hash changed'):
        validate_saved_run(rows,info,run)


def test_reference_and_raw_outputs_are_escaped_and_replay_is_not_a_new_run(completed):
    from scripts.launch_verification_gradio import validate_saved_run, case_guide, decision_cards, result_message
    info, rows, run = completed
    summary, predictions, calls = validate_saved_run(rows,info,run)
    rows[0]['question'] = '<script>bad()</script>'
    predictions[0]['raw'] = '<img src=x onerror=bad()>'
    predictions[0]['error'] = 'timeout'
    predictions[0]['parsed'] = None
    guide = case_guide(rows,'Test')
    cards = decision_cards(rows,predictions,calls)
    assert '<script>' not in guide and '&lt;script&gt;' in guide
    assert '<img src=x' not in cards and '&lt;img' in cards
    assert 'Runtime error' in cards
    assert 'No new model calls' in result_message(summary, archived=True)
    assert 'No new model calls' not in result_message(summary)
    assert '100%' not in cards.split('</style>', 1)[1]


def test_builtin_gradio_locale_is_english_without_changing_other_headers():
    import asyncio
    from scripts.launch_verification_gradio import EnglishLocaleMiddleware
    observed = []
    async def app(scope, receive, send): observed.append(scope)
    middleware = EnglishLocaleMiddleware(app)
    original = {'type':'http', 'headers':[(b'accept-language',b'zh-CN'),(b'x-test',b'keep')]}
    asyncio.run(middleware(original,None,None))
    assert observed[0]['headers'] == [(b'x-test',b'keep'),(b'accept-language',b'en')]
    assert original['headers'][0][1] == b'zh-CN'
    websocket = {'type':'websocket','headers':[]}
    asyncio.run(middleware(websocket,None,None))
    assert observed[-1] is websocket


def test_english_locale_precedes_client_boot_and_preserves_non_html():
    import asyncio
    from scripts.launch_verification_gradio import EnglishLocaleMiddleware
    async def run(content_type, chunks):
        sent = []
        async def app(scope, receive, send):
            await send({'type':'http.response.start', 'status':200,
                        'headers':[(b'content-type',content_type), (b'content-length',b'1')]})
            for i, chunk in enumerate(chunks):
                await send({'type':'http.response.body', 'body':chunk,
                            'more_body':i < len(chunks)-1})
        async def send(message): sent.append(message)
        await EnglishLocaleMiddleware(app)({'type':'http','headers':[]},None,send)
        return sent
    sent = asyncio.run(run(b'text/html', [b'<html><he', b'ad><script type="module">boot()</script></head></html>']))
    body = sent[1]['body']
    assert body.index(b'Object.defineProperty(navigator') < body.index(b'type="module"')
    assert int(dict(sent[0]['headers'])[b'content-length']) == len(body)
    assert len(sent) == 2 and sent[1]['more_body'] is False
    sent = asyncio.run(run(b'application/json', [b'{"ok":', b'true}']))
    assert [part['body'] for part in sent[1:]] == [b'{"ok":', b'true}']
