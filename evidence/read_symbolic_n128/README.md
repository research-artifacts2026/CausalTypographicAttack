# Retrospective transcription-plus-symbolic baseline

This analysis reuses all 512 independent-read responses from the two-checkpoint,
128-item media study. **It makes zero new model calls.** The checker receives
only actual model text and the public task assumption, never gold fields, item
IDs, family labels, generator parameters or target answers. Known record schemas
and explicit numerical relations are programmed in advance of this scoring pass.
This is a retrospective, template-aware baseline, not an originally registered
defense confirmation or a general scientific verifier.

| Model | Direct correct pairs | Read + rules correct pairs | Covered records |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | 3/128 | 124/128 | 249/256 |
| Qwen3-VL-8B | 8/128 | 127/128 | 254/256 |

All abstentions count incorrect. Every covered record agrees with nominal gold.
That does not certify the data as independently human validated. Unit tests use
separate synthetic fixtures. The reference-text sanity check is separate from
model scoring. Missing or conflicting fields abstain; no parser repair was made
after scoring these archived responses.

Reads used a 384-token cap and direct judgments used 96. The comparison is not
compute-matched. Reused scenes, known grammar and post hoc analysis limit transfer
claims. Two exploratory paired tests use source-stratified bootstrap intervals
and Holm correction; see `analysis.json` and the frozen `protocol.json`.

From the repository root:

```bash
python -m unittest discover -s tests -p test_transcribed_record_checker.py -v
python scripts/evaluate_transcribed_record_checker.py --output evidence/read_symbolic_n128 --replay
```

Replay also requires the unchanged sibling `evidence/channel_binding_n128`
directory. Input hashes and executed checker/analyzer snapshots are retained.
`predictions.jsonl` preserves every extracted field and abstention reason.
