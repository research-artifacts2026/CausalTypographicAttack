#!/usr/bin/env python3
"""Gradio interface for the paper's controlled verification protocol."""
from __future__ import annotations
import argparse
import json
import os
import sys
import uuid
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cta.verification_workbench import (GOLD, STRATEGIES, archive_session, canonical, evaluate_packet,
    freeze_packet, load_packet, read_jsonl, score, sha)
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
            return '攻击成功：模型把无效记录判为有效'
        if row['accuracy']['record_false']['k'] == 1:
            return '攻击未成功：模型正确识别无效记录'
        return '攻击未达目标：模型误判为无记录'
    return f"攻击成功 {row['dc_asr']['k']}/{row['dc_asr']['n']} 个对照合格样本"


def result_message(summary):
    lines = ['### 本次结果', f"已完成 {summary['actual_calls']} 次模型调用；运行错误 {summary['failed_calls']} 次。"]
    for name, row in summary['strategies'].items():
        lines.append(f"- **{LABELS[name]}：{outcome(row)}**")
    lines.append('\nPASS / FAIL 表示该项判断正确 / 错误；EOR failure 的 YES 表示在合格条件下仍接受了无效记录。'
                 '独立 Read/Know 在每个样本上只运行一组，并由各策略共享，不是各策略内部完成了读取和规则推理。')
    return '\n\n'.join(lines)


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
        gallery = [(r["image_path"], {"source_absent": "Source · 原图", "record_true": "Valid · 有效记录", "record_false": "Invalid · 无效记录"}[r["condition"]]) for r in frozen]
        calls = sum(6 if s == "read_then_verify" else 3 for s in strategies) + 2
        text = (f"### Packet frozen · 已冻结\n**{calls} model calls** for this item; no feedback search.\n\n"
                "The source and valid controls determine eligibility. Full-set errors are still retained.\n\n"
                f"**Same decision query across all three states:**\n\n{frozen[0]['question']}\n\n"
                f"**Assumptions (rule-guided arm only):** {frozen[0]['record']['assumption']}")
        return str(packet), gallery, text, info, [], None, [], '### 尚未运行：先构建样本，再运行验证', []

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
        return (metrics_table(summary), summary, str(bundle), answers_table(frozen, predictions),
                result_message(summary), shared_probe_table(predictions, read_jsonl(run_root/'calls.jsonl')))

    def replay(session_id):
        if session_id not in {value for _,value in saved_choices}:
            raise gr.Error('Select an available completed run.')
        session = (output_root / session_id).resolve()
        if not session.is_relative_to(output_root.resolve()):
            raise gr.Error('Saved run outside this workbench.')
        packet = session / 'packet'
        info, frozen = load_packet(packet)
        run_root = session / 'evaluation'
        summary = json.loads((run_root/'summary.json').read_text(encoding='utf-8'))
        if sha(run_root/'calls.jsonl') != summary['call_log_sha256']:
            raise gr.Error('Saved call journal hash changed; audit required.')
        predictions = read_jsonl(run_root/'predictions.jsonl')
        replayed = score(frozen,predictions,info['strategies'])
        if any(replayed[key] != summary[key] for key in replayed):
            raise gr.Error('Saved prediction scores differ from summary; audit required.')
        gallery = [(r['image_path'], r['condition']) for r in frozen]
        bundle = session/'evaluation.zip'
        return (None, gallery, '### Archived run / 已有结果回看\n没有新增模型调用。\n\n'+frozen[0]['question'],
                summary, metrics_table(summary), str(bundle) if bundle.exists() else None,
                answers_table(frozen,predictions), result_message(summary),
                shared_probe_table(predictions,read_jsonl(run_root/'calls.jsonl')))

    with gr.Blocks(title="ContraLedger · Verification Lab", fill_width=True) as demo:
        gr.Markdown("# ContraLedger\n### Verification Lab · 视觉记录验证实验室\nRead the fields. Check the relation. Inspect every control.")
        gr.Markdown("Compare **source → valid → invalid** using a frozen question and answer map. "
                    "This workbench is aligned with the paper's three-state task; additional prompting strategies are exploratory diagnostics.")
        gr.Markdown("**Single illustrative frozen item; not an aggregate estimate.**\n\n"
                    "**单个冻结样本示例，不是总体实验估计。** 转录辅助判断是把模型转录结果交给模型再次判断；"
                    "它不执行符号计算，也不是论文中的 **Read + rules symbolic checker**。")
        state = gr.State()
        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                mode = gr.Radio(["Frozen example / 已有样本", "Upload image / 上传图片"],
                    value="Frozen example / 已有样本" if choices else "Upload image / 上传图片", label="Input source")
                example = gr.Dropdown(choices, value=choices[0][1] if choices else None, label="Frozen sample")
                with gr.Group(visible=not bool(choices)) as upload_group:
                    image = gr.Image(type="filepath", label="Uploaded scene", height=220, sources=["upload"])
                    label = gr.Textbox(label="Visible object for upload", placeholder="e.g. refrigerator")
                    family = gr.Dropdown(list(REQUESTED_COUNTERFACTUAL_FAMILIES), value="unit_conversion", label="Constraint family for upload")
                strategies = gr.CheckboxGroup([(LABELS[s], s) for s in STRATEGIES], value=list(STRATEGIES), label="Decision prompting strategies / 判断提示策略（调用预算不同）")
                prepare_button = gr.Button("1 · Prepare & freeze / 构建并冻结", variant="secondary")
                run_button = gr.Button("2 · Evaluate / 运行验证", variant="primary")
                with gr.Accordion('Inspect a completed run / 回看已有结果（不调用模型）', open=False):
                    saved_run = gr.Dropdown(saved_choices, value=saved_choices[0][1] if saved_choices else None, label='Saved run')
                    replay_button = gr.Button('Load saved results / 加载已有结果')
                gr.Markdown("Changing inputs takes effect after **Prepare & freeze**. Models never choose or alter the frozen records. "
                            "Rule-guided supplies the same assumption for both twins; transcription-assisted decision consumes additional calls and adds no executable checker.")
            with gr.Column(scale=3, min_width=400):
                gallery = gr.Gallery(label="The three states · click an image to inspect", columns=3, rows=1, object_fit="contain", height=330, interactive=False)
                status = gr.Markdown("### Ready · 请先选择样本或上传图片")
        verdict = gr.Markdown('### 尚未运行：先构建样本，再运行验证')
        gr.Markdown('### Shared independent probes / 样本级共享探测\n'
                    '每个样本仅一组独立 Read + Know（共两次调用），各策略共享。'
                    '它们与转录辅助策略自身的转录调用不同；Know 保留原图，不是纯文本基线。')
        probes = gr.Dataframe(headers=['样本','共享 probe','结果','输入形式','实际 prompt','Raw output','运行错误'],
                              interactive=False, wrap=True, label='独立探测原始记录（不按策略重复计数）')
        answers = gr.Dataframe(headers=['策略','图片状态','模型原始答案','模型判断','正确答案','判断是否正确'], interactive=False, wrap=True, label='逐张图片核对：模型究竟回答了什么')
        results = gr.Dataframe(headers=["策略", "本次攻击判定", "原图判断", "有效记录判断", "无效记录判断", "双记录均正确", "对照通过", "EOR eligible / 合格", "EOR failure / 条件下误接受", "未解析答案数"],
            interactive=False, wrap=True, label="单样本结果：PASS/FAIL 为判断正确性；YES/NO 为事件是否发生")
        download = gr.File(label="Download images, exact prompts and raw responses")
        with gr.Accordion("Audit details / 完整核验信息", open=False):
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
        prepare_button.click(prepare, [mode, example, image, label, family, strategies], [state, gallery, status, details, results, download, answers, verdict, probes])
        run_button.click(run, [state], [results, details, download, answers, verdict, probes])
        replay_button.click(replay, [saved_run], [state, gallery, status, details, results, download, answers, verdict, probes])
        mode.change(lambda value: gr.update(visible=value == "Upload image / 上传图片"), [mode], [upload_group], queue=False)
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
