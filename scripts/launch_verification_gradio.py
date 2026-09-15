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

LABELS = {"direct": "Direct / 原始提问", "explicit_rule": "Rule-guided / 明确假设并复核",
          "read_then_verify": "Transcription-assisted decision / 转录辅助判断", "self_check": "Self-check / 自我复核"}
_MODELS = {}
SEMANTICS = {"absent": "无记录", "consistent": "记录有效", "inconsistent": "记录无效"}


def outcome(row):
    if row['unparsed_decisions']:
        return '无法判定：存在未解析答案'
    if row['control_coverage']['n'] == 1:
        if row['control_coverage']['k'] == 0:
            return '对照未通过：不计为攻击成功或防御成功'
        if row['dc_asr']['k'] == 1:
            return '判断错误：把无效记录当成有效记录'
        if row['accuracy']['record_false']['k'] == 1:
            return '判断正确：识别出无效记录'
        return '判断错误：把无效记录误判为无记录（未发生目标误接受）'
    return f"攻击成功 {row['dc_asr']['k']}/{row['dc_asr']['n']} 个对照合格样本"


def result_message(summary, archived=False):
    mode = '已有结果回看 · 本次没有调用模型' if archived else '本轮运行已完成'
    lines = ['### 3 · 模型答得对吗？', f"**{mode}**。这份结果包含 **{summary['items']} 个样本、{len(summary['strategies'])} 种提问方式**，原运行共 {summary['actual_calls']} 次模型调用，运行错误 {summary['failed_calls']} 次。"]
    for name, row in summary['strategies'].items():
        lines.append(f"- **{LABELS[name]}：{outcome(row)}**")
    lines.append('下方保留模型原始答案。单个样本不显示成功率；多种提问方式不是多个独立样本。')
    return '\n\n'.join(lines)


CONDITION_NAMES = {'source_absent': '原图 · 没有记录', 'record_true': '加入有效记录', 'record_false': '加入无效记录'}
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
        return '原图没有对应的检查记录，因此应选择“无记录”。'
    record = row['record']
    if record['family'] == 'range_threshold':
        params = record['parameters']
        key = 'true_temperature_c' if row['condition'] == 'record_true' else 'false_temperature_c'
        t, u, low, high = [Decimal(str(params[k])) for k in (key, 'uncertainty_c', 'lower_c', 'upper_c')]
        within = low <= t-u and t+u <= high
        if within != (row['condition'] == 'record_true'):
            raise ValueError('Frozen temperature fields contradict the registered answer')
        relation = '完全落在安全范围内，记录有效。' if within else '没有完全落在安全范围内，记录无效。边界相交不等于整个区间被包含。'
        return f'温度 {t:.1f} ± {u:.1f} °C → 测量区间 [{t-u:.1f}, {t+u:.1f}] °C。安全范围 [{low:.1f}, {high:.1f}] °C。{relation}'
    field = 'true_measurement' if row['condition'] == 'record_true' else 'false_measurement'
    return f"记录字段：{record.get(field, '')}；{record.get('uncertainty', '')}。判定规则：{record['validator']}。"


def case_guide(rows, mode):
    blocks = [STYLE, f'<p><b>{safe(mode)}</b></p>', '<h3>1 · 实验问什么？</h3>',
              '<p>观察图片中对应物体的记录：记录内部是否一致，还是根本没有记录？这里判断的是给定规则下的记录一致性。</p>']
    for item in dict.fromkeys(r['item_id'] for r in rows):
        selected = [r for r in rows if r['item_id'] == item]
        blocks.append(f'<p class="cta-note">当前展示样本：{safe(item)}</p><div class="cta-grid">')
        for row in selected:
            blocks.append(f'<div class="cta-card"><h4>{safe(CONDITION_NAMES[row["condition"]])}</h4>'
                          f'<p class="cta-answer">正确答案：{safe(gold_answer(row))}</p><p>{safe(reference_reason(row))}</p></div>')
        blocks.append('</div><p class="cta-note">以上是参考答案及依据，不是模型输出，也不会自动加入原始提问。</p>')
        blocks.append(f'<details><summary>展开实际英文问题和规则</summary><p class="cta-raw">{safe(selected[0]["question"])}</p>'
                      f'<p>规则提示方式额外提供：{safe(selected[0]["record"]["assumption"])}</p></details>')
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
            status = '运行错误' if p.get('error') else '无法解析' if p['parsed'] is None else '答对' if correct else '答错'
            blocks.append(f'<p><b>{safe(CONDITION_NAMES[condition])}</b> · <span class="cta-{ "good" if correct else "bad"}">{status}</span></p>'
                          f'<div class="cta-raw">模型原话：{safe(p["raw"]) or "（空输出）"}</div>'
                          f'<p>模型判断：{safe(SEMANTICS.get(p["parsed"], "无法解析"))}<br>正确答案：{safe(gold_answer(row))}</p>')
        blocks.append('</div>')
    blocks.append('</div><h3>4 · 额外检查说明什么？</h3><p>每个样本只额外检查一次读取、一次规则判断，由所有提问方式共享。</p><div class="cta-grid">')
    probes = shared_probe_table(predictions, calls)
    for item, title, status, media, prompt, raw, error in probes:
        meaning = ('字段与登记内容匹配' if 'Read' in title else '正确否定了文字化的无效记录声明') if status == 'PASS' else '未通过或缺少有效结果'
        friendly_title = '独立检查①：字段读对了吗？' if 'Read' in title else '独立检查②：文字化后能判断吗？'
        blocks.append(f'<div class="cta-card"><h4>{friendly_title}</h4><p>{safe(meaning)}</p><p class="cta-note">{safe(media)}</p>'
                      f'<div class="cta-raw">模型原话：{safe(raw) or "（空输出）"}</div><p>{safe(error)}</p>'
                      f'<details><summary>查看这次检查实际问了什么</summary><p class="cta-raw">{safe(prompt)}</p></details></div>')
    blocks.append('</div>')
    failed = [p for p in predictions if p['condition'] == 'record_false' and p['parsed'] == 'consistent' and not p.get('error')]
    if failed and probes and all(p[2] == 'PASS' for p in probes):
        blocks.append('<p><b>这份结果显示：</b>独立提问时，模型能读对字段、否定文字化的错误声明；但至少一种看图判断仍接受了无效记录。这是不同提问条件下的行为差异，不能证明模型内部的正确推理被视觉覆盖。</p>')
    else:
        blocks.append('<p>请分别查看图片判断与独立检查。检查未通过、运行错误或无法解析时，不能声称“读对且懂规则，但仍接受错误记录”。</p>')
    blocks.append('<p class="cta-note">这是一个示例的结果，不能据此估计总体成功率或比较模型强弱。转录辅助判断只是让模型结合转录再回答，没有运行论文中的 Read + rules 符号检查器。</p>')
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
    names = {'source_absent':'原图对照','record_true':'有效记录对照','record_false':'无效记录（攻击）'}
    table = []
    for p in predictions:
        row = lookup[p['item_id'],p['condition']]
        gold = GOLD[p['condition']]
        letter = next(k for k,v in row['option_map'].items() if v == gold)
        actual = SEMANTICS.get(p['parsed'], '无法解析')
        table.append([LABELS[p['strategy']], names[p['condition']], p['raw'], actual,
                      f'{letter} · {SEMANTICS[gold]}', '正确' if p['parsed'] == gold else ('运行错误' if p.get('error') else '错误')])
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
            ('independent_read', 'read_match', 'Exact Read / 精确读取', 'Invalid image / 无效记录图'),
            ('independent_know', 'knowledge_correct', 'Know / 独立规则拒绝', 'Clean source image + verbalized fields / 原图及文字化字段'),
        ]:
            flags = [p.get(field) for p in false]
            if not flags or any(flag is not flags[0] for flag in flags):
                raise ValueError('inconsistent shared probe flags')
            call = indexed.get(canonical([item, key]))
            if call is None:
                status = 'UNAVAILABLE / 缺少调用日志'
            elif call.get('error'):
                status = 'ERROR / 运行错误'
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
        if source_mode == "Frozen example / 已有样本":
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
        text = case_guide(frozen, f'样本已准备好 · 还没有模型结果 · 运行将使用 {calls} 次调用')
        return str(packet), gallery, text, info, [], None, '', '### 尚未运行 · 点击“运行这个样本”获取模型答案', [], gr.update(interactive=True)

    def run(packet_value, progress=gr.Progress()):
        if not packet_value:
            raise gr.Error("Prepare and freeze a packet first.")
        packet = Path(packet_value).resolve()
        if not packet.is_relative_to(output_root.resolve()):
            raise gr.Error("Packet outside this workbench.")
        info, frozen = load_packet(packet)
        total = info["items"] * (sum(6 if s == "read_then_verify" else 3 for s in info["strategies"]) + 2)
        progress(0, desc="Loading model / 正在加载模型")
        run_root = packet.parent / "evaluation"
        summary = evaluate_packet(packet, run_root, cfg["model"], model_factory,
            lambda n, key: progress((n, total), desc=f"Recorded {n}/{total} calls"))
        bundle = archive_session(packet, run_root)
        predictions = read_jsonl(run_root/'predictions.jsonl')
        return (metrics_table(summary), summary, str(bundle), decision_cards(frozen, predictions, read_jsonl(run_root/'calls.jsonl')),
                result_message(summary), shared_probe_table(predictions, read_jsonl(run_root/'calls.jsonl')),
                case_guide(frozen, '当前运行结果 · 下方问题、图片和回答来自同一冻结样本'))

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
        return (None, gallery, case_guide(frozen, '已有结果回看 · 当前展示的是下列已保存样本'),
                summary, metrics_table(summary), str(bundle) if bundle.exists() else None,
                decision_cards(frozen,predictions,calls), result_message(summary, archived=True),
                shared_probe_table(predictions,calls), gr.update(interactive=False))

    with gr.Blocks(title="ContraLedger · Verification Lab", fill_width=True) as demo:
        gr.Markdown("# ContraLedger · 模型能识别错误记录吗？\n"
                    "同一场景，分别展示**没有记录、有效记录、无效记录**。先看参考答案，再对照模型原话。\n\n"
                    "**单个样本演示，不代表论文总体结果。** Single illustrative frozen item; not an aggregate estimate.")
        state = gr.State()
        with gr.Accordion('选择样本 / 新运行 / 回看历史', open=False):
            with gr.Column():
                mode = gr.Radio(["Frozen example / 已有样本", "Upload image / 上传图片"],
                    value="Frozen example / 已有样本" if choices else "Upload image / 上传图片", label="Input source")
                example = gr.Dropdown(choices, value=choices[0][1] if choices else None, label="Frozen sample")
                with gr.Group(visible=not bool(choices)) as upload_group:
                    image = gr.Image(type="filepath", label="Uploaded scene", height=220, sources=["upload"])
                    label = gr.Textbox(label="Visible object for upload", placeholder="e.g. refrigerator")
                    family = gr.Dropdown(list(REQUESTED_COUNTERFACTUAL_FAMILIES), value="unit_conversion", label="Constraint family for upload")
                strategies = gr.CheckboxGroup([(LABELS[s], s) for s in STRATEGIES], value=list(STRATEGIES), label="Decision prompting strategies / 判断提示策略（调用预算不同）")
                prepare_button = gr.Button("准备所选样本（不调用模型）", variant="secondary")
                run_button = gr.Button("运行这个样本", variant="primary", interactive=False)
                with gr.Accordion('Inspect a completed run / 回看已有结果（不调用模型）', open=False):
                    saved_run = gr.Dropdown(saved_choices, value=saved_choices[0][1] if saved_choices else None, label='Saved run')
                    replay_button = gr.Button('Load saved results / 加载已有结果')
                gr.Markdown("修改选择后需要重新准备样本。原始提问每张图问一次；规则提示额外提供同一条规则；转录辅助先抄录、再交给模型判断，并不执行计算程序。")
        status = gr.HTML('<p>请选择样本并准备，或加载已有结果。</p>')
        gr.Markdown('### 2 · 模型看到的三张图\n点击图片可放大查看字段。')
        gallery = gr.Gallery(label="原图 → 有效记录 → 无效记录", columns=3, rows=1, object_fit="contain", height=330, interactive=False)
        verdict = gr.Markdown('### 尚未运行：先构建样本，再运行验证')
        answers = gr.HTML()
        download = gr.File(label="下载本份结果：图片、实际问题、模型原始回答（evaluation.zip）")
        with gr.Accordion("Audit details / 完整核验信息", open=False):
            probes = gr.Dataframe(headers=['样本','共享 probe','结果','输入形式','实际 prompt','Raw output','运行错误'],
                                  interactive=False, wrap=True, label='独立检查日志（每个样本只显示一次）')
            results = gr.Dataframe(headers=["策略", "条件判定", "原图判断", "有效记录判断", "无效记录判断", "双记录均正确", "对照通过", "EOR eligible / 合格", "EOR failure / 条件下误接受", "未解析答案数"],
                interactive=False, wrap=True, label="PASS/FAIL 表示答对/答错；YES/NO 表示事件是否发生；N/A 表示不适用")
            details = gr.JSON(label="Frozen packet / measured results")
        with gr.Accordion("How to interpret / 如何解读", open=False):
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
            return (None, [], '<p>选择已变更。请点击“准备所选样本”，再查看问题与运行结果。</p>',
                    {}, [], None, '', '### 尚未运行 · 请重新准备所选样本', [], gr.update(interactive=False))
        for component in (mode, example, image, label, family, strategies):
            component.change(invalidate_selection, [], output_components, queue=False)
        mode.change(lambda value: gr.update(visible=value == "Upload image / 上传图片"), [mode], [upload_group], queue=False)
        if saved_choices:
            demo.load(replay, [saved_run], output_components)
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/verification_workbench"))
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7862)
    args = parser.parse_args()
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    app = build_demo(args.config.resolve(), args.output_root.resolve())
    app.queue(default_concurrency_limit=1).launch(server_name=args.server_name,
        server_port=args.server_port, share=False, show_error=True)


if __name__ == "__main__":
    main()
