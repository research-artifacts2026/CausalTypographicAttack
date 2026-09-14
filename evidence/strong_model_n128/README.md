# Qwen3.5-27B: completed larger-model replication

All 1,280 new calls completed on the same 128 scene items and two validity states
used in the preceding symbolic confirmation. Zero runtime errors. This is new
checkpoint inference on previously evaluated scenes, not new-task validation.

| Arm | Output cap | Pair correct / 128 | Unparsed / 256 | Cap hits |
|---|---:|---:|---:|---:|
| Direct, non-thinking | 384 | 2 | 47 | 8 |
| Reasoned, non-thinking | 384 | 76 | 54 | 54 |
| Read + fixed rules, non-thinking | 384 | 103 | 39 | 0 |
| Longer reasoned, non-thinking | 2048 | 113 | 5 | 5 |
| Thinking enabled | 4096 | 91 | 42 | 42 |

Pair correctness requires both valid and false states correct. Every parsing
failure and checker abstention remains wrong in the primary scores. Thinking
token counts include the reasoning output. The same frozen prompts and checker
are used throughout; the last two arms explicitly have different budgets.

Read-plus-rules beats the two 384-token neural arms after Holm correction across
all four registered contrasts. It does not establish an advantage over longer
reasoning (numerically 103 versus 113, corrected p = 0.252694) or thinking
(103 versus 91, corrected p = 0.252694). All unparsed reasoned/thinking outputs
hit their caps. Optimistically crediting unparsed answers yields reasoned bounds
of 122/128, 118/128 and 127/128 respectively. Thus the short-cap gain does not
establish superior intrinsic reasoning or a general defense advantage.

All 217 covered read records match nominal gold; 39 abstentions remain. The
descriptive format audit includes an omitted SAFE RANGE field example, showing
that transcription field coverage is a limitation of the fixed pipeline.

## Replay without a model

From the repository root:
```
python scripts/analyze_strong_model.py --replay evidence/strong_model_n128
python scripts/audit_strong_model_formats.py --evidence evidence/strong_model_n128
python -m unittest discover -s tests -p test_strong_model.py -v
```

`calls.jsonl` retains final text, reasoning text, complete API JSON, requests,
token usage and timestamps. `predictions.jsonl` retains all verdicts and checker
abstentions. `provenance.json` binds the original analyzer outputs. Supplemental
source, runtime, upstream-identity and format audits are also included; they do
not change the frozen study. All exported bytes are bound by the anonymous
package's separate manifest.

## Model identity and repeating inference

The official checkpoint revision is
`fc05daec18b0a78c049392ed2e771dde82bdf654` from
[Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B/tree/fc05daec18b0a78c049392ed2e771dde82bdf654).
All 21 checkpoint files match that immutable upstream revision; all 45 executed
project source files match pre-result public commit `1ed050a`.
The original environment used vLLM 0.19.1, Transformers 5.14.1, PyTorch
2.10.0+cu129 and two RTX A6000 GPUs. Full versions are in `runtime_audit.json`.

The reference launcher below was added for release usability after registration;
the original `study_design.json` records the actual original service arguments.
On Linux, with the checkpoint already downloaded and two idle 48 GB GPUs:
```
python scripts/launch_strong_model_service.py --checkpoint /path/to/Qwen3.5-27B --vllm-bin /path/to/vllm --output runs/strong-service --gpus 0,1
```
Wait for the local service's `/health` endpoint before running. Register a new,
unique run with `register_strong_model.py`, then execute `run_strong_model.py`.
Registration requires the original registered scene packet and its licensed
image assets; this text-only export does not bundle COCO/VOC photographs or
model weights. Obtain those assets under their original terms. Numerical replay
works from the released responses without either dependency.

This is a greedy controlled protocol, not the provider's recommended sampled
maximum-performance configuration. Human gold validation, closed-model/GPT
replication and new-schema transfer remain unestablished. These scores are
verification accuracy, not a new measurement of the primary three-state DC-ASR.
