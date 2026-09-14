# Scene-disjoint mitigation confirmation: no confirmed advantage

Both checkpoints completed all 3,584 registered calls, with zero runtime errors
or unparsed decisions. The 128 scenes comprise 64 COCO and 64 VOC photographs,
eight per source/family. They exclude the earlier 64-scene mitigation diagnostic
by item ID and exact original image SHA256. They remain previously used benchmark
images; this is not globally unseen-source transfer. No outcome informed selection.

The two fixed strategies receive identical three-state images, questions and
answer maps. Both use a 384-token draft/transcription cap and a 96-token final
cap per state with greedy decoding; actual token lengths are not matched.
Six calls per strategy plus two shared Read/Know probes yield 14 calls per item
and model. Two computational shards per model were merged only after completion.

Primary endpoint: both valid and invalid record decisions correct, on all scenes.
Two-sided exact paired McNemar tests use Holm over the two model comparisons.
Confidence intervals use 10,000 source-stratified paired bootstrap resamples,
with seed 20260914. The exploratory diagnostic is not pooled into confirmation.

| Model | Self-check pair correct | Read then verify | Difference, pp | 95% CI, pp | Holm p |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 1/128 | 2/128 | +0.78125 | [-1.5625, 3.90625] | 1 |
| Qwen3-VL-8B | 12/128 | 21/128 | +7.03125 | [-0.78125, 14.84375] | 0.244156 |

Discordant pairs are 2 versus 1 for Qwen-7B and 18 versus 9 for Qwen3-VL.
The earlier Qwen3-VL diagnostic signal is not confirmed by this registered test.
This does not establish equivalence or absence of an effect. Full-set valid
accuracy on Qwen3-VL decreases from 101/128 with self-check to 78/128 with
read-then-verify, despite improved invalid-record rejection. Conditional attack
rates therefore require their exact control denominators and are descriptive.
The evidence does not establish a reliable defense, an internal mechanism, or
superiority of one attack format.

```bash
python scripts/replay_verification_confirmation.py --evidence evidence/verification_confirmation_n128
```

The portable replay checks all raw calls, decision parsing, Read/Know probes,
metric denominators, paired vectors, intervals and Holm correction. The separate
original-server audit also checks frozen image hashes, registration timing,
execution identities, duplicate exclusion and exact per-shard coverage.
`study_design.json` replaces private checkpoint paths with model names while
retaining the original registration hash. Photographs and weights are external.
