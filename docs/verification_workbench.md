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
2. Choose the evaluation strategies and press **Prepare & freeze**.
3. Inspect all three images and the identical decision question.
4. Press **Evaluate** and download the complete packet and raw call journal.

Changing a widget does not change a frozen packet: prepare again before a new
experiment. Successful or failed old runs are retained in their own folders.

## Strategies and budgets

| Strategy | Intervention | Calls per triplet |
|---|---|---:|
| Direct | Original frozen decision query | 3 |
| Rule-guided | Same truth-independent assumptions and checking instruction for each state | 3 |
| Read then verify | Independent model transcription for each state, followed by a decision using the image and quoted transcription | 6 |
| Shared diagnostic probes | Exact false-record transcription and verbalized-rule rejection with the clean source image | 2 |

All strategies together use **14 calls per item**, not equal compute. These are
diagnostic interventions, not a budget-matched algorithm ranking. No strategy
receives a hidden answer, target label, symbolic residual, or oracle transcript.
The rule-guided arm receives the registered assumptions, not an instance answer.
Its name does not imply it uses tools or a symbolic verifier.

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

31 targeted CPU tests pass, including the unchanged three-state and legacy UI
regressions. The old legacy-config assertion was corrected to accept its already
supported `auto_scene` setting. An eight-family Qwen2.5-VL-7B integration run
completed **112 actual calls**, with zero runtime failures or unparsed decisions.
This verifies the execution path; it is not a new attack-effectiveness estimate.

The separately released `evidence/rule_explicit_confirmation_n128/` records a
previously completed and freshly replayed **different** study: 1,280 calls,
Qwen-7B-specific gain and no net Qwen3-VL gain. It does not establish general
superiority for the workbench, the original ContraLedger protocol, or a defense.
