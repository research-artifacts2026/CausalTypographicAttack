# Prospective symbolic-checker confirmation

Completed 2026-09-14: 1,536 fresh calls, 128 archived scene items, two states,
two checkpoints, three arms. Excludes the prior 128-item retrospective and
64-item diagnostic populations by item ID and original-source SHA256. These
are different archived scenes, not globally unseen sources or new schemas.
Registration preceded inference; checkpoint files and executed source were hashed.

| Model | Direct pairs | Reason then answer | Read + fixed rules |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | 2/128 | 41/128 | 120/128 |
| Qwen3-VL-8B | 10/128 | 93/128 | 124/128 |

All arms use one call with 384 maximum output tokens. Actual input/output
lengths differ and authored checker rules add task knowledge. All four
prespecified exact paired contrasts have Holm p < .0001. Missing final-answer
format affects 36/256 and 21/256 reasoned records; the latter model hits the
output cap 21 times. Zero runtime errors. All abstentions count wrong.
An explicitly post hoc upper bound credits every unparsed reasoned record as
correct: at most 66/128 and 110/128 pairs, still below the fixed-checker counts.
This does not estimate performance at a larger budget or revise primary scores.

Replay from the code root:
```
python scripts/analyze_symbolic_confirmation.py --replay evidence/symbolic_confirmation_n128
python scripts/make_symbolic_confirmation_tests.py --evidence evidence/symbolic_confirmation_n128
python scripts/symbolic_confirmation_sensitivity.py --evidence evidence/symbolic_confirmation_n128
```

Raw outputs, all predictions, exact prompts and token traces are retained.
`provenance.json` binds the original analyzer outputs; the test-table provenance
binds its later display. The sensitivity file is a labeled post hoc bound.
COCO/VOC photographs and model weights remain external under original terms.
Independent human gold validation, new-schema transfer, closed-model replication
and a general defense claim are not established by this study.
