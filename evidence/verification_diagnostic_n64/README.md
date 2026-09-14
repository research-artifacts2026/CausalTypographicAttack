# Registered mitigation diagnostic on reused scenes

64 archived scenes: 32 COCO and 32 VOC, four scenes per source-family cell.
Two models completed all 2,560 calls. Selection was fixed by item-ID hash and
family round-robin before inference; no outcomes were used to select items.

Primary endpoint: all-scene valid-invalid pair accuracy, read-then-verify versus
self-check. Both arms use two calls per state with the same 384-token first-stage
and 96-token final-stage caps. Actual lengths and input-token counts may differ.
Exact paired McNemar tests use Holm across the two model contrasts; paired
source-stratified bootstrap intervals use 10,000 draws and seed 20260914.

| Model | Self-check pair correct | Read then verify | Difference | Holm p |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 0/64 | 1/64 | +1.5625 pp | 1 |
| Qwen3-VL-8B | 6/64 | 16/64 | +15.625 pp | 0.0425415 |

This is partial, checkpoint-dependent mitigation on reused sources. It does
not establish a reliable defense, new-source transfer, neural mechanism, or
attack superiority. All four strategies and exact conditional denominators
are retained in `generated_table.tex` and `analysis.json`.

```bash
python scripts/replay_verification_diagnostic.py --evidence evidence/verification_diagnostic_n64
```

This portable replay checks all raw decision responses, parsing, metrics,
paired vectors, intervals and multiple-testing correction. Pixel hashes,
registration timing and original execution identities were separately checked
by `scripts/analyze_verification_diagnostic.py` on the complete frozen packet.
The released call journals do not include source photographs or model weights.
