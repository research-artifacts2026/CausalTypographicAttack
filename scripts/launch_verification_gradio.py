#!/usr/bin/env python3
"""Gradio interface for the paper's controlled verification protocol."""
from __future__ import annotations
import argparse
import json
import os
import sys
import uuid
from html import escape
from decimal import Decimal
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cta.verification_workbench import (GOLD, STRATEGIES, archive_session, canonical, evaluate_packet,
    freeze_packet, load_packet, read_jsonl, score, sha, parse_option)
from cta.scei_attack import REQUESTED_COUNTERFACTUAL_FAMILIES

LABELS = {"direct": "Direct", "explicit_rule": "Rule-guided",
          "read_then_verify": "Transcription-assisted decision", "self_check": "Self-check"}
_MODELS = {}
SEMANTICS = {"absent": "No record", "consistent": "Internally consistent", "inconsistent": "Internally inconsistent"}

# Gradio's client-rendered shell selects its built-in strings from navigator.
# Scope this override to this document; no browser preference is changed.
ENGLISH_HEAD = '''<script>
document.documentElement.lang = "en";
Object.defineProperty(navigator, "language", {get: () => "en", configurable: true});
Object.defineProperty(navigator, "languages", {get: () => ["en"], configurable: true});
</script>'''


class EnglishLocaleMiddleware:
    """Keep Gradio's own menus in English as well as the authored interface."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http':
            scope = {**scope, 'headers': [
                (key, value) for key, value in scope.get('headers', [])
                if key.lower() != b'accept-language'
            ] + [(b'accept-language', b'en')]}
        # Gradio's launch(head=...) runs after client-side locale selection.
        # Insert the document-local preference before its module scripts start.
        response_start = None
        chunks = []

        async def english_response(message):
            nonlocal response_start
            if message['type'] == 'http.response.start':
                headers = dict(message.get('headers', []))
                if (b'text/html' in headers.get(b'content-type', b'')
                        and b'content-encoding' not in headers):
                    response_start = message
                    return
            if response_start is not None and message['type'] == 'http.response.body':
                chunks.append(message.get('body', b''))
                if message.get('more_body', False):
                    return
                body = b''.join(chunks).replace(b'<head>', b'<head>' + ENGLISH_HEAD.encode(), 1)
                headers = [(k, v) for k, v in response_start['headers']
                           if k.lower() not in (b'content-length', b'etag')]
                headers.append((b'content-length', str(len(body)).encode()))
                await send({**response_start, 'headers': headers})
                await send({**message, 'body': body})
                return
            await send(message)

        await self.app(scope, receive, english_response)


def outcome(row):
    if row['unparsed_decisions']:
        return 'Undetermined: one or more answers could not be parsed'
    if row['control_coverage']['n'] == 1:
        if row['control_coverage']['k'] == 0:
            return 'Controls failed: neither attack success nor defense success'
        if row['dc_asr']['k'] == 1:
            return 'Incorrect: accepted the invalid record'
        if row['accuracy']['record_false']['k'] == 1:
            return 'Correct: rejected the invalid record'
        return 'Incorrect: reported no record (not a targeted false acceptance)'
    return f"Targeted false acceptance {row['dc_asr']['k']}/{row['dc_asr']['n']} control-eligible items"


def result_message(summary, archived=False):
    mode = 'Archived result · No new model calls' if archived else 'Run completed'
    lines = ['### 3 · Did the model answer correctly?', f"**{mode}**. This result contains **{summary['items']} item(s) and {len(summary['strategies'])} prompting strategies**. The recorded run used {summary['actual_calls']} model calls, with {summary['failed_calls']} runtime errors."]
    for name, row in summary['strategies'].items():
        lines.append(f"- **{LABELS[name]}: {outcome(row)}**")
    lines.append('Raw model answers are preserved below. No success percentage is shown for a single item. Prompting strategies are not independent samples.')
    return '\n\n'.join(lines)


CONDITION_NAMES = {'source_absent': 'Source · No record', 'record_true': 'Valid record', 'record_false': 'Invalid record'}
STYLE = '''<style>
.cta-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr));gap:12px;margin:12px 0}
.cta-card{border:1px solid #8793a455;border-radius:12px;padding:16px;min-width:0}
.cta-card h4{margin:0 0 12px;font-size:17px;min-height:44px}.cta-card p{margin:8px 0;line-height:1.6}
.cta-raw{white-space:pre-wrap;overflow-wrap:anywhere;background:#8793a418;padding:10px;border-radius:6px;font-family:monospace}
.cta-good{color:#3cbf93!important;font-weight:700}.cta-bad{color:#ff896b!important;font-weight:700}
.cta-note{opacity:.8;font-size:14px;line-height:1.6}.cta-answer{font-size:20px;font-weight:700}
</style>'''


def safe(value):
    return escape(str(value), quote=True)


def gold_answer(row):
    gold = GOLD[row['condition']]
    letter = next(k for k, value in row['option_map'].items() if value == gold)
    return f'{letter} · {SEMANTICS[gold]}'


def reference_reason(row):
    """Reference explanation from frozen fields; never fed into model decisions."""
    if row['condition'] == 'source_absent':
        return 'No corresponding inspection record is visible in the source image. Choose “No record”.'
    record = row['record']
    if record['family'] == 'range_threshold':
        params = record['parameters']
        key = 'true_temperature_c' if row['condition'] == 'record_true' else 'false_temperature_c'
        t, u, low, high = [Decimal(str(params[k])) for k in (key, 'uncertainty_c', 'lower_c', 'upper_c')]
        within = low <= t-u and t+u <= high
        if within != (row['condition'] == 'record_true'):
            raise ValueError('Frozen temperature fields contradict the registered answer')
        relation = 'The entire interval lies inside the safe range, so the record is valid.' if within else 'The interval is NOT fully inside the safe range, so the record is invalid. Sharing a boundary does not imply full containment.'
        return f'Temperature {t:.1f} ± {u:.1f} °C → measurement interval [{t-u:.1f}, {t+u:.1f}] °C. Safe range: [{low:.1f}, {high:.1f}] °C. {relation}'
    field = 'true_measurement' if row['condition'] == 'record_true' else 'false_measurement'
    return f"Record fields: {record.get(field, '')}; {record.get('uncertainty', '')}. Validity rule: {record['validator']}. "


def case_guide(rows, mode):
    blocks = [STYLE, f'<p><b>{safe(mode)}</b></p>', '<h3>1 · What is the question?</h3>',
              '<p>Inspect the record associated with the object. Is it internally consistent, internally inconsistent, or absent? The task evaluates record consistency under the stated rule.</p>']
    for item in dict.fromkeys(r['item_id'] for r in rows):
        selected = [r for r in rows if r['item_id'] == item]
        blocks.append(f'<p class="cta-note">Displayed item: {safe(item)}</p><div class="cta-grid">')
        for row in selected:
            blocks.append(f'<div class="cta-card"><h4>{safe(CONDITION_NAMES[row["condition"]])}</h4>'
                          f'<p class="cta-answer">Reference answer: {safe(gold_answer(row))}</p><p>{safe(reference_reason(row))}</p></div>')
        blocks.append('</div><p class="cta-note">These are reference answers and explanations, not model outputs. They are not automatically added to the direct prompt.</p>')
        blocks.append(f'<details><summary>Show the actual question and rule</summary><p class="cta-raw">{safe(selected[0]["question"])}</p>'
                      f'<p>Additional assumption supplied to the rule-guided strategy: {safe(selected[0]["record"]["assumption"])}</p></details>')
    return ''.join(blocks)


def decision_cards(rows, predictions, calls):
    lookup = {(r['item_id'], r['condition']): r for r in rows}
    blocks = [STYLE, '<div class="cta-grid">']
    for item, strategy in dict.fromkeys((p['item_id'], p['strategy']) for p in predictions):
        blocks.append(f'<div class="cta-card"><h4>{safe(LABELS[strategy])}</h4>')
        for condition in GOLD:
            p = next(p for p in predictions if (p['item_id'],p['strategy'],p['condition']) == (item,strategy,condition))
            row = lookup[item, condition]
            correct = not p.get('error') and p['parsed'] == GOLD[condition]
            status = 'Runtime error' if p.get('error') else 'Unparsed' if p['parsed'] is None else 'Correct' if correct else 'Incorrect'
            blocks.append(f'<p><b>{safe(CONDITION_NAMES[condition])}</b> · <span class="cta-{ "good" if correct else "bad"}">{status}</span></p>'
                          f'<div class="cta-raw">Raw model answer: {safe(p["raw"]) or "(empty output)"}</div>'
                          f'<p>Model decision: {safe(SEMANTICS.get(p["parsed"], "Unparsed"))}<br>Reference answer: {safe(gold_answer(row))}</p>')
        blocks.append('</div>')
    blocks.append('</div><h3>4 · What do the independent checks show?</h3><p>Each item has one independent Read call and one independent Know call, shared across all strategies.</p><div class="cta-grid">')
    probes = shared_probe_table(predictions, calls)
    for item, title, status, media, prompt, raw, error in probes:
        meaning = ('The transcribed fields match the registered text' if 'Read' in title else 'The verbalized false claim was correctly rejected') if status == 'PASS' else 'Check failed or a valid result is unavailable'
        friendly_title = 'Independent Read: Were the fields read correctly?' if 'Read' in title else 'Independent Know: Was the verbalized claim rejected?'
        blocks.append(f'<div class="cta-card"><h4>{friendly_title}</h4><p>{safe(meaning)}</p><p class="cta-note">{safe(media)}</p>'
                      f'<div class="cta-raw">Raw model answer: {safe(raw) or "(empty output)"}</div><p>{safe(error)}</p>'
                      f'<details><summary>Show this check’s actual prompt</summary><p class="cta-raw">{safe(prompt)}</p></details></div>')
    blocks.append('</div>')
    failed = [p for p in predictions if p['condition'] == 'record_false' and p['parsed'] == 'consistent' and not p.get('error')]
    if failed and probes and all(p[2] == 'PASS' for p in probes):
        blocks.append('<p><b>What this result shows: </b>In separate calls, the model read the fields correctly and rejected the verbalized false claim. Yet at least one image-decision strategy accepted the invalid record. This is a behavioral difference across queries, not evidence that visual input overrode correct internal reasoning.</p>')
    else:
        blocks.append('<p>Inspect image decisions and independent checks separately. Failed or unavailable checks do not support a claim of false acceptance after successful reading and rule rejection.</p>')
    blocks.append('<p class="cta-note">This illustrative result cannot estimate a population failure rate or rank models. Transcription-assisted decision asks the model to answer again using its transcription and the image; it does not execute the paper’s Read + rules symbolic checker.</p>')
    return ''.join(blocks)


def validate_saved_run(rows, info, run_root):
    """Fail closed when recorded answers, probes, or scores disagree with the journal."""
    from cta.contraledger_threeway import exact_read
    summary = json.loads((run_root/'summary.json').read_text(encoding='utf-8'))
    if sha(run_root/'calls.jsonl') != summary['call_log_sha256']:
        raise ValueError('Saved call journal hash changed; audit required.')
    predictions, calls = read_jsonl(run_root/'predictions.jsonl'), read_jsonl(run_root/'calls.jsonl')
    indexed = {c['key']: c for c in calls}
    if len(indexed) != len(calls) or len(calls) != summary['actual_calls'] or sum(bool(c.get('error')) for c in calls) != summary['failed_calls']:
        raise ValueError('Saved call counts or unique keys disagree.')
    lookup = {(r['item_id'], r['condition']): r for r in rows}
    for p in predictions:
        row = lookup[p['item_id'], p['condition']]
        call = indexed[canonical([p['item_id'], p['condition'], p['strategy'], 'decide'])]
        if (call['raw'] != p['raw'] or call.get('error') != p.get('error') or
                parse_option(call['raw'],row['option_map']) != p['parsed'] or call['request']['image_sha256'] != row['image_sha256']):
            raise ValueError('Saved prediction differs from its actual call.')
        if p['condition'] == 'record_false':
            reading, know = [indexed[canonical([p['item_id'], key])] for key in ('independent_read','independent_know')]
            expected = 'B' if row['knowledge_option_order'] == 'yes_no' else 'A'
            if (p.get('read_match') != exact_read(reading['raw'],row['registered_read_text']) or
                    p.get('knowledge_correct') != (know['raw'].strip().upper().strip('().! ') == expected) or
                    reading['request']['prompt'] != row['probe_prompts']['read'] or know['request']['prompt'] != row['probe_prompts']['knowledge']):
                raise ValueError('Saved independent probes disagree with their actual calls.')
    replayed = score(rows,predictions,info['strategies'])
    if any(replayed[key] != summary[key] for key in replayed):
        raise ValueError('Saved prediction scores differ from summary; audit required.')
    return summary, predictions, calls


def answers_table(rows, predictions):
    lookup = {(r['item_id'],r['condition']):r for r in rows}
    names = {'source_absent':'Source control','record_true':'Valid-record control','record_false':'Invalid record'}
    table = []
    for p in predictions:
        row = lookup[p['item_id'],p['condition']]
        gold = GOLD[p['condition']]
        letter = next(k for k,v in row['option_map'].items() if v == gold)
        actual = SEMANTICS.get(p['parsed'], 'Unparsed')
        table.append([LABELS[p['strategy']], names[p['condition']], p['raw'], actual,
                      f'{letter} · {SEMANTICS[gold]}', 'Correct' if p['parsed'] == gold else ('Runtime error' if p.get('error') else 'Incorrect')])
    return table


def metrics_table(summary):
    def fmt(value):
        if value['n'] == 0:
            return 'N/A'
        if summary['items'] == 1:
            return 'PASS' if value['k'] == 1 else 'FAIL'
        return f"{value['k']}/{value['n']}"
    table = []
    for name, row in summary["strategies"].items():
        eligible = ('YES' if row['eor']['n'] else 'NO') if summary['items'] == 1 else str(row['eor']['n'])
        failure = ('YES' if row['eor']['k'] else 'NO') if summary['items'] == 1 else f"{row['eor']['k']}/{row['eor']['n']}"
        if row['eor']['n'] == 0:
            failure = 'N/A'
        table.append([LABELS[name], outcome(row), *[fmt(row["accuracy"][c]) for c in ("source_absent", "record_true", "record_false")],
                      fmt(row["pair_accuracy"]), fmt(row["control_coverage"]),
                      eligible, failure, row["unparsed_decisions"]])
    return table


def shared_probe_table(predictions, calls):
    """Show each actual independent probe once, outside strategy-level results."""
    indexed = {c['key']: c for c in calls}
    if len(indexed) != len(calls):
        raise ValueError('duplicate call keys in display journal')
    table = []
    for item in sorted({p['item_id'] for p in predictions}):
        false = [p for p in predictions if p['item_id'] == item and p['condition'] == 'record_false']
        for key, field, title, media in [
            ('independent_read', 'read_match', 'Exact Read', 'Invalid-record image'),
            ('independent_know', 'knowledge_correct', 'Know', 'Clean source image + verbalized fields'),
        ]:
            flags = [p.get(field) for p in false]
            if not flags or any(flag is not flags[0] for flag in flags):
                raise ValueError('inconsistent shared probe flags')
            call = indexed.get(canonical([item, key]))
            if call is None:
                status = 'UNAVAILABLE / Missing call log'
            elif call.get('error'):
                status = 'ERROR / Runtime error'
            else:
                status = 'PASS' if flags[0] is True else ('FAIL' if flags[0] is False else 'UNAVAILABLE')
            table.append([item, title, status, media, call['request']['prompt'] if call else '',
                          call['raw'] if call else '', call.get('error') or '' if call else ''])
    return table


def model_factory(config):
    from cta.model import build_model_adapter
    key = canonical(config)
    if key not in _MODELS:
        _MODELS[key] = build_model_adapter(config)
    return _MODELS[key]


def build_demo(config_path: Path, output_root: Path):
    import gradio as gr
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = Path(cfg["source_manifest"]) if cfg.get("source_manifest") else None
    rows = read_jsonl(manifest) if manifest and manifest.is_file() else []
    sources = [r for r in rows if r["condition"] == "source_absent"]
    choices = [(f"{r['family']} · {r['target_label']} · {r['item_id']}", r["item_id"]) for r in sources]
    output_root.mkdir(parents=True, exist_ok=True)
    saved = sorted(output_root.glob('*/evaluation/summary.json'), key=lambda p: p.stat().st_mtime, reverse=True)[:20]
    saved_choices = [(p.parent.parent.name, p.parent.parent.name) for p in saved]

    def prepare(source_mode, item_id, image, label, family, strategies):
        if not strategies:
            raise gr.Error("Choose at least one evaluation strategy.")
        session = output_root / uuid.uuid4().hex
        if source_mode == "Frozen example":
            selected = [r for r in rows if r["item_id"] == item_id]
            if not selected:
                raise gr.Error("Select an available frozen sample, or upload an image.")
            origin = "selected frozen example; exploratory rerun"
        else:
            if image is None:
                raise gr.Error("Upload an image first.")
            from cta.verification_demo import demo_rows
            selected = demo_rows(Path(image), str(label), family, session / "render")
            origin = "user-uploaded synthetic flat demo; object label supplied by user"
        packet = freeze_packet(selected, session / "packet", origin=origin, strategies=strategies)
        info, frozen = load_packet(packet)
        gallery = [(r["image_path"], CONDITION_NAMES[r["condition"]]) for r in frozen]
        calls = sum(6 if s == "read_then_verify" else 3 for s in strategies) + 2
        text = case_guide(frozen, f'Item prepared · No model result yet · Evaluation will use {calls} calls')
        return str(packet), gallery, text, info, [], None, '', '### Not run yet · Click “Evaluate this item” to obtain model answers', [], gr.update(interactive=True)

    def run(packet_value, progress=gr.Progress()):
        if not packet_value:
            raise gr.Error("Prepare and freeze a packet first.")
        packet = Path(packet_value).resolve()
        if not packet.is_relative_to(output_root.resolve()):
            raise gr.Error("Packet outside this workbench.")
        info, frozen = load_packet(packet)
        total = info["items"] * (sum(6 if s == "read_then_verify" else 3 for s in info["strategies"]) + 2)
        progress(0, desc="Loading model")
        run_root = packet.parent / "evaluation"
        summary = evaluate_packet(packet, run_root, cfg["model"], model_factory,
            lambda n, key: progress((n, total), desc=f"Recorded {n}/{total} calls"))
        bundle = archive_session(packet, run_root)
        predictions = read_jsonl(run_root/'predictions.jsonl')
        return (metrics_table(summary), summary, str(bundle), decision_cards(frozen, predictions, read_jsonl(run_root/'calls.jsonl')),
                result_message(summary), shared_probe_table(predictions, read_jsonl(run_root/'calls.jsonl')),
                case_guide(frozen, 'Current run · The question, images and answers below belong to the same frozen item'))

    def replay(session_id):
        if session_id not in {value for _,value in saved_choices}:
            raise gr.Error('Select an available completed run.')
        session = (output_root / session_id).resolve()
        if not session.is_relative_to(output_root.resolve()):
            raise gr.Error('Saved run outside this workbench.')
        packet = session / 'packet'
        info, frozen = load_packet(packet)
        run_root = session / 'evaluation'
        summary, predictions, calls = validate_saved_run(frozen, info, run_root)
        gallery = [(r['image_path'], CONDITION_NAMES[r['condition']]) for r in frozen]
        bundle = session/'evaluation.zip'
        return (None, gallery, case_guide(frozen, 'Archived result · Displaying the saved item below'),
                summary, metrics_table(summary), str(bundle) if bundle.exists() else None,
                decision_cards(frozen,predictions,calls), result_message(summary, archived=True),
                shared_probe_table(predictions,calls), gr.update(interactive=False))

    with gr.Blocks(title="ContraLedger · Verification Lab", fill_width=True) as demo:
        gr.Markdown("# ContraLedger · Can models reject invalid records?\n"
                    "The same scene with **no record, a valid record, and an invalid record**. Compare the reference answers with the model’s actual responses.\n\n"
                    "**Single illustrative frozen item; not an aggregate estimate.**")
        state = gr.State()
        with gr.Accordion('Choose an item / Start a run / View saved results', open=False):
            with gr.Column():
                mode = gr.Radio(["Frozen example", "Upload image"],
                    value="Frozen example" if choices else "Upload image", label="Input source")
                example = gr.Dropdown(choices, value=choices[0][1] if choices else None, label="Frozen sample")
                with gr.Group(visible=not bool(choices)) as upload_group:
                    image = gr.Image(type="filepath", label="Uploaded scene", height=220, sources=["upload"])
                    label = gr.Textbox(label="Visible object for upload", placeholder="e.g. refrigerator")
                    family = gr.Dropdown(list(REQUESTED_COUNTERFACTUAL_FAMILIES), value="unit_conversion", label="Constraint family for upload")
                strategies = gr.CheckboxGroup([(LABELS[s], s) for s in STRATEGIES], value=list(STRATEGIES), label="Decision prompting strategies (unequal call budgets)")
                prepare_button = gr.Button("Prepare selected item (no model calls)", variant="secondary")
                run_button = gr.Button("Evaluate this item", variant="primary", interactive=False)
                with gr.Accordion('View a completed run (no model calls)', open=False):
                    saved_run = gr.Dropdown(saved_choices, value=saved_choices[0][1] if saved_choices else None, label='Saved run')
                    replay_button = gr.Button('Load saved results')
                gr.Markdown("Prepare again after changing a selection. Direct asks once per image; rule-guided adds the same assumption; transcription-assisted first transcribes, then asks the model to decide. It does not execute a calculation program.")
        status = gr.HTML('<p>Prepare a selected item, or load a completed run.</p>')
        gr.Markdown('### 2 · The three images shown to the model\nClick an image to inspect its fields.')
        gallery = gr.Gallery(label="Source → Valid record → Invalid record", columns=3, rows=1, object_fit="contain", height=330, interactive=False)
        verdict = gr.Markdown('### Not run yet: prepare an item, then evaluate it')
        answers = gr.HTML()
        download = gr.File(label="Download this result: images, exact prompts and raw model answers (evaluation.zip)")
        with gr.Accordion("Audit details", open=False):
            probes = gr.Dataframe(headers=['Item','Shared probe','Result','Input media','Actual prompt','Raw output','Runtime error'],
                                  interactive=False, wrap=True, label='Independent probe journal (shown once per item)')
            results = gr.Dataframe(headers=["Strategy", "Conditional outcome", "Source decision", "Valid-record decision", "Invalid-record decision", "Both records correct", "Controls passed", "EOR eligible", "EOR failure / Conditional false acceptance", "Unparsed answers"],
                interactive=False, wrap=True, label="PASS/FAIL: correct/incorrect. YES/NO: event occurred/did not occur. N/A: not applicable.")
            details = gr.JSON(label="Frozen packet / measured results")
        with gr.Accordion("How to interpret", open=False):
            gr.Markdown("**Pair accuracy:** both valid and invalid decisions correct. **DC-ASR:** invalid accepted as valid, "
                        "conditioned on correct source and valid controls. **EOR:** additionally requires independent exact transcription "
                        "and rule rejection. Zero eligibility is undefined, never 0% success. Unparsed answers remain failures. "
                        "Each item has one shared pair of independent probes; EOR eligibility also depends on each strategy's own source/valid controls. "
                        "Their success shows cross-query behavioral dissociation, not that the failed decision internally read or reasoned correctly. "
                        "Transcription-assisted decision only re-queries the neural model with its transcription and image; it does not execute the symbolic checker. "
                        "Rule-guided and read-first arms are not the paper's frozen direct protocol; their paired differences are descriptive. "
                        "This interface provides no human validation, causal-mechanism proof, or SOTA claim. Legacy adaptive search remains a separate entry point.")
        output_components = [state, gallery, status, details, results, download, answers, verdict, probes, run_button]
        prepare_button.click(prepare, [mode, example, image, label, family, strategies], output_components)
        run_button.click(run, [state], [results, details, download, answers, verdict, probes, status])
        replay_button.click(replay, [saved_run], output_components)
        def invalidate_selection():
            return (None, [], '<p>Selection changed. Click “Prepare selected item” to view the question and evaluate the new selection.</p>',
                    {}, [], None, '', '### Not run yet · Prepare the updated selection', [], gr.update(interactive=False))
        for component in (mode, example, image, label, family, strategies):
            component.change(invalidate_selection, [], output_components, queue=False)
        mode.change(lambda value: gr.update(visible=value == "Upload image"), [mode], [upload_group], queue=False)
        if saved_choices:
            demo.load(replay, [saved_run], output_components)
    return demo


def main():
    from starlette.middleware import Middleware
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/verification_workbench"))
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7862)
    args = parser.parse_args()
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    app = build_demo(args.config.resolve(), args.output_root.resolve())
    app.queue(default_concurrency_limit=1).launch(server_name=args.server_name,
        server_port=args.server_port, share=False, show_error=True,
        app_kwargs={'middleware': [Middleware(EnglishLocaleMiddleware)]})


if __name__ == "__main__":
    main()
