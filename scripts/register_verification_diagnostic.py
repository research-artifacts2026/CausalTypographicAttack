"""Freeze a bounded, outcome-oblivious two-source strategy diagnostic."""
import argparse
import json
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cta.verification_workbench import freeze_packet, read_jsonl, select_items, sha, now, source_code_hashes, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--coco', type=Path, required=True)
    p.add_argument('--voc', type=Path, required=True)
    p.add_argument('--qwen7', type=Path, required=True)
    p.add_argument('--qwen3', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    rows = select_items(read_jsonl(a.coco), 32) + select_items(read_jsonl(a.voc), 32)
    strategies = ['direct', 'explicit_rule', 'read_then_verify', 'self_check']
    packet = freeze_packet(rows, a.output / 'packet', origin='Prospectively frozen strategy diagnostic on reused archived sources, not an unseen-scene confirmation', strategies=strategies)
    configs = {}
    for name, path in [('qwen7',a.qwen7), ('qwen3',a.qwen3)]:
        cfg = yaml.safe_load(path.read_text())['model']
        checkpoint = Path(cfg['name_or_path'])
        configs[name] = {'config':cfg, 'checkpoint_config_sha256':sha(checkpoint/'config.json'),
                         'checkpoint_revision':checkpoint.name if checkpoint.parent.name == 'snapshots' else None}
        (a.output / (name+'.local.yaml')).write_text(yaml.safe_dump({'model':cfg}))
    write_json(a.output/'registration.json', {
        'registered_at':now(), 'items':64, 'source_balance':{'coco':32,'voc':32},
        'selection':'32 per source; stable item-ID SHA256 order round-robin over eight families; no outcome input',
        'strategies':strategies,'models':configs,'expected_calls_per_model':1280,'expected_total_calls':2560,
        'packet_sha256':sha(packet/'packet.json'),'code':source_code_hashes(),
        'primary_endpoint':'All-scene valid-invalid pair accuracy, read_then_verify minus self_check, separately for both models',
        'primary_test':'Two-sided exact paired McNemar; Holm across the two registered model contrasts',
        'ci':'10000 paired source-stratified bootstrap resamples, seed 20260914',
        'secondary_endpoints':'All-state accuracy, balanced true/false accuracy, control coverage, DC-ASR and EOR with exact denominators; descriptive only',
        'budget':'direct=3; rule=3; read_then_verify=6; self_check=6; shared Read/Know=2 calls per triplet. Two-stage arms both allow 384 draft/transcription + 96 final tokens per state; actual token lengths need not match.',
        'stopping':'Every registered cell once; retain errors/unparsed answers, no adaptive selection, no significance-based stopping or silent retry',
        'scope':'Post-hoc mitigation diagnostic on reused scenes. This does not establish fresh-source transfer, neural mechanism or attack SOTA.'})
    print(json.dumps({'packet':str(packet),'items':64,'expected_calls':2560}))


if __name__ == '__main__': main()
