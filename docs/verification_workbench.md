# ContraLedger Verification Lab

The paper's main task is source/valid/invalid verification. The new Gradio
workbench implements that task without the feedback search used by the legacy
SCEI demo. A run is an exploratory diagnostic unless separately registered.

## Start

Use a dedicated environment with the model dependencies for your checkpoint:

```bash
pip install -r requirements-workbench.txt
# Download the checkpoint first: the existing adapter uses local_files_only.
# Copy configs/verification_workbench.yaml and point name_or_path at it.
python scripts/launch_verification_gradio.py --config configs/my_workbench.yaml
```

Open http://127.0.0.1:7862. For a remote server, forward that port over SSH.
The launcher defaults to loopback and does not create a public sharing link.
Do not expose model execution or research-image downloads publicly without
appropriate access controls. No API credentials are needed for a local model.

The default config supports image uploads. To browse pre-rendered historical
samples, set `source_manifest` to a three-state JSONL manifest whose assets exist
on the host. Historical source files are copied into a new immutable packet,
never edited. User-upload examples use a full-ink flat footer with shared font
size/geometry and a user-supplied object label; this is an exploratory digital
carrier, not automatic scene grounding or a new population benchmark.

1. Select an example or upload an image and name the visible object.
2. Choose the evaluation strategies and press **Prepare selected item (no model calls)**.
3. Inspect all three images and the identical decision question.
4. Press **Evaluate this item** and download the complete packet and raw call journal.

Changing a widget does not change a frozen packet: prepare again before a new
experiment. Successful or failed old runs are retained in their own folders.

The page follows question → reference answers → actual images → model answers
→ shared diagnostics. Each decision card preserves the raw response and labels
it correct, incorrect, unparsed, or a runtime error. Reference calculations are
display-only: they are not model reasoning and are not injected into prompts.
The temperature example explains full interval containment, including why a
shared boundary is insufficient. Detailed conditional metrics and logs are
collapsed under the audit section; a single item is not presented as a success
percentage.

If completed runs exist, opening the page automatically replays the newest one
without inference, with an explicit archived-result label. Open the sample
controls to choose a different archived run or prepare a new item. Changing
inputs disables execution until the new selection is prepared. Archived replay
validates the frozen assets, call-journal hash, raw/parsed predictions, shared
probe outcomes, call counts and recomputed scores before displaying results.

## Strategies and budgets

| Strategy | Intervention | Calls per triplet |
|---|---|---:|
| Direct | Original frozen decision query | 3 |
| Rule-guided | Same truth-independent assumptions and checking instruction for each state | 3 |
| Transcription-assisted decision | Model transcription for each state, followed by another neural decision using the image and quoted transcription; no executable rule checker | 6 |
| Shared diagnostic probes | Exact false-record transcription and verbalized-rule rejection with the clean source image | 2 |

All strategies together use **14 calls per item**, not equal compute. These are
diagnostic interventions, not a budget-matched algorithm ranking. No strategy
receives a hidden answer, target label, symbolic residual, or oracle transcript.
The rule-guided arm receives the registered assumptions, not an instance answer.
Its name does not imply it uses tools or a symbolic verifier.

The UI labels the historical `read_then_verify` strategy as
**Transcription-assisted decision**. The internal ID is retained
for compatibility with frozen packets and archived analyses; its prompts,
budgets and scoring are unchanged. Its second stage does not add the family
assumption or run interval arithmetic. It is distinct from the paper's
**Read + rules** baseline, which applies executable schema rules to a model
transcription. Neither label implies that a neural decision reasoned correctly.

The page shows **Single illustrative frozen item; not an aggregate estimate**.
Single-item control/decision correctness uses PASS/FAIL, with the actual raw
answer and registered answer map alongside it. Independent Read and Know appear
once in **Shared independent probes**, including their exact prompts and raw
outputs. They are different calls from the transcription-assisted arm's reads.
Each strategy shows its own EOR eligibility and conditional false-acceptance
event (YES/NO; N/A when ineligible), not a repeated 100% estimate. Numeric
aggregate definitions remain available unchanged in the downloadable summary.

The completed-run viewer checks the frozen packet, call-journal hash and
prediction-to-summary agreement before displaying saved results. It makes no
new model calls. Independent probe success and image-decision failure establish
a cross-query behavioral dissociation; they do not prove that correct internal
reasoning was overridden. Uniform acceptance whenever a record is present is
consistent with a record-presence bias, but one example cannot identify a
general heuristic or mechanism.

## Metrics and failure handling

- Full-set source, valid and invalid accuracies retain all selected scenes.
- Pair accuracy requires both valid and invalid decisions to be correct.
- Balanced validity accuracy averages valid/invalid correctness.
- DC-ASR requires correct source and valid controls; control coverage is shown.
- EOR additionally requires an independent exact read and correct Know rejection.
  Know retains the clean source image: it is **not an image-free text baseline**.
- Comparisons use the intersection of each pair of strategies' eligible scenes
  and retain paired discordants. No statistical superiority is asserted.
- Empty eligibility is `null`, never zero success. Malformed answers and runtime
  failures remain visible. The strict complete-option parser is versioned and
  never matches an incidental article "a". Historical parsers/scores are unchanged.

Packets hash every image and manifest. Resume is refused if the packet, model
config, package source or runtime identity changes. Each completed call is
fsynced. A pending call at interruption is marked unknown and blocks automatic
resume; it must not be retried and counted as if no call occurred. Failed calls
are retained without silent retries. A per-run lock prevents duplicate workers.
Model provenance is stored alongside the final journal hash.

## Batch CLI

```bash
python scripts/run_verification_workbench.py freeze \
  --manifest /path/to/threeway/manifest.jsonl --items 8 \
  --output runs/verification_smoke/packet
python scripts/run_verification_workbench.py verify --packet runs/verification_smoke/packet
python scripts/run_verification_workbench.py run \
  --packet runs/verification_smoke/packet --output runs/verification_smoke/evaluation \
  --config configs/my_workbench.yaml
```

Selection is a stable ID-hash ordering round-robin across families, without model
outputs. Reusing old scenes is a smoke test or diagnostic, not held-out evidence.
Real fresh-study registration and independent human review remain separate work.

## Validation on 2026-09-14

The UI now gives an explicit per-item attack outcome and a raw-answer/correct-
answer table. Invalid-record accuracy of zero can mean attack success; DC-ASR
is labeled from the attacker's perspective. A failed control or unparsed answer
is not shown as a successful defense. The corrected UI was verified with 14
fresh real-model calls on the same example, without selecting a different image.

The CLI additionally supports `self_check`, a two-stage draft/review baseline.
It has the same per-stage output caps and call count as `read_then_verify`.
The frozen 64-scene, two-model diagnostic is released in
`evidence/verification_diagnostic_n64/`; all 2,560 calls completed. It is a
post-hoc mitigation study on reused sources, not a fresh-source attack benchmark.
The registered comparison shows a Qwen3-VL-specific pair-accuracy improvement,
with substantial residual failures and no significant Qwen-7B improvement.
42 targeted tests pass; the identity-scrubbed candidate export also passed 72
workbench, display, table-replay and archived-experiment tests plus 11 subtests.

Validated environment: Python 3.10, torch 2.5.1+cu121, transformers 5.9.0,
Gradio 6.26.0. Historical warning messages are retained in the test record.

31 targeted CPU tests pass, including the unchanged three-state and legacy UI
regressions. The old legacy-config assertion was corrected to accept its already
supported `auto_scene` setting. An eight-family Qwen2.5-VL-7B integration run
completed **112 actual calls**, with zero runtime failures or unparsed decisions.
This verifies the execution path; it is not a new attack-effectiveness estimate.

The separately released `evidence/rule_explicit_confirmation_n128/` records a
previously completed and freshly replayed **different** study: 1,280 calls,
Qwen-7B-specific gain and no net Qwen3-VL gain. It does not establish general
superiority for the workbench, the original ContraLedger protocol, or a defense.
