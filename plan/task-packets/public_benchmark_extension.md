# Public benchmark extension task packet

## Goal

Test Evidence-CTA under a public, question-conditioned protocol without
claiming that incomparable PPIA, REALM, RIO-Bench, and RVTA ASR values share a
denominator or threat model.

## Frozen public sources

| Source | Local commit | Role | Current boundary |
|---|---|---|---|
| RIO-Bench | `2425419ddb5b7247121290f7ead5b4cd137f1a55` | Primary public read-or-ignore protocol and official scorer | Dataset is CC-BY-4.0; code repository has no top-level license file in this checkout. |
| SceneTAP | `cd2b72285b424ca674ae2ec2c05b2c55291613c7` | Full scene-coherent baseline | Requires SoM, TextDiffuser-2, and a multimodal planner; checkout has no top-level license file. |
| REALM | `e3aea21b9e0c0c4e2bec80c128221ac297fe5c69` | Separate NIPS2017/ImageNet red-team track | Its 12-attack ASR is not directly comparable with VQA clean-conditioned ASR; checkout has no top-level license file. |
| PPIA | `7291d92ee05c4b9079ef824e0b4bd12e8e14316c` | Physical-environment external validation target | README says the code is still being organized and cannot yet run completely. The repository license file is Apache-2.0 despite an MIT README badge. |

## Registered RIO Obj-MC pilot

- Split: official `val`.
- Selection: 100 question IDs with smallest `SHA256(seed:question_id)`, seed
  20260824, resolved at a recorded Hugging Face dataset revision.
- Official conditions: clean plus RIO easy/medium/hard typography.
- Added matched conditions: naive typography, in-house scene-coherent plaque,
  direct causal claim, and Evidence-CTA.
- Target: the official hard attack word when it corresponds to a wrong MC
  option; deterministic wrong option otherwise.
- Queries: one original question per condition, no response-adaptive search.
- Primary metric: official RIO Obj-MC accuracy and clean-conditioned ASR.
- Secondary: targeted ASR and exact paired McNemar tests.
- Expansion gate: run 100 first; expand to 300--500 only after complete logs,
  valid provenance, and no condition-pairing failures.

## Simulated and real physical tracks

- Simulated capture profiles use deterministic perspective, illumination,
  Gaussian blur, downsampling, and JPEG degradation. They must be reported as
  simulated camera degradation.
- Real physical validation requires printed/displayed attacks, a fixed camera
  protocol, documented distances/angles/lighting, and raw photographs. No
  simulated frame may enter the physical table.

## SceneTAP decision

The current Hugging Face dataset card lists eleven RIO configurations and does
not list the `*__scenetap` variants mentioned by the construction README.
Therefore, no precomputed SceneTAP number is assumed available. A full
SceneTAP row is admitted only after either (a) the public variant is located
and pinned or (b) the official planner is run with its required dependencies.
The in-house plaque remains labeled in-house.

## Human evaluation gate

Three independent annotators must complete the existing randomized forms.
The analyzer requires all three response files and rejects incomplete packs.
No model-generated or author-filled response substitutes for independent
human judgments.

## Paper claims allowed before new runs

- The public protocol adapter and simulated-capture generator are implemented
  and locally tested.
- No RIO, REALM, PPIA, full SceneTAP, or physical ASR may be reported until
  complete raw logs exist.
- Existing RVTA numbers remain internal benchmark results, not public SOTA.
