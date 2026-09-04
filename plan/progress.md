# RVTA-Bench implementation progress

## ContraLedger truth-by-cue identification (2026-09-04)

- [x] Define the fixed one-image/one-question 2 x 3 truth-by-cue protocol.
- [x] Preserve values-only, non-evaluative authority, and explicit-conclusion conditions.
- [x] Add deterministic builder, three-probe runner, EOR analysis, and unit tests.
- [x] Materialize and visually audit the balanced 120-item development manifest on server 212.
- [x] Complete four-model Qwen2.5-VL-3B/7B, LLaVA-OneVision-1.5-8B, and InternVL2-8B held-out jobs and source-prior diagnostics.
- [x] Freeze a disjoint 400-item held-out expansion before inspecting development victim results (zero source-ID overlap).
- [x] Add manifest/image/mask hash checks, frozen-input equality checks, polarity splits, and exact paired cue tests to the analyzer.
- [x] Add results to the manuscript only after run-completeness and hash audits pass.

- [x] Freeze scope, partitions, conditions, labels, and prohibited claims.
- [x] Implement exact-area and benign-evidence controls.
- [x] Implement the disjoint held-out factorial split.
- [x] Implement truth-aware transfer metrics and evidence generation.
- [x] Add model configurations and annotation validation.
- [x] Run tests and manifest smoke checks for the original RVTA pipeline.
- [x] Run matched and factorial inference.
- [x] Generate completed evidence assets.
- [x] Revise and push the paper.

## Public benchmark extension

- [x] Freeze RIO-Bench, SceneTAP, REALM, and PPIA public-code commits and threat-model boundaries.
- [x] Implement RIO Obj-MC materialization, question-conditioned CTA generation, official-score replay, and deterministic simulated capture.
- [x] Register Qwen-3B/7B, LLaVA-OneVision-8B, and InternVL2-8B configs and strict run-completeness audit.
- [x] Materialize the official RIO validation blocks on server 212.
- [x] Run Qwen-3B/7B and replay the official RIO scorer after completeness checks.
- [x] Run LLaVA/InternVL transfer under the identical manifests.
- [x] Complete the SoM--local-Qwen-planner--TextDiffuser diagnostic; this is not official GPT-4o-planner equivalence.
- [ ] Complete real physical capture and three-person independent human evaluation.

## RVTA-QA balanced-v1 and AI-natural carrier (2026-08-25)

- [x] Implement truth-polarity, A/B-order, and AB-versus-YES/NO counterbalancing.
- [x] Score correct and target responses semantically per item.
- [x] Preserve fixed paired questions and independent claim-transcription audits.
- [x] Add COCO/VOC configurations for four open LVLM checkpoints.
- [x] Add build, run, validation, CSV/JSON, and LaTeX generation scripts.
- [x] Add six-cell unit tests and run the local smoke suite.
- [x] Generate three provenance-locked AI natural-carrier examples.
- [x] Prepare a blinded three-person naturalness pack with hidden repeats.
- [x] Materialize the two 300-item manifests on server 212.
- [x] Run eight complete model-dataset jobs and validate hashes.
- [x] Run the four-model n=3 natural-carrier pilot (qualitative only).
- [ ] Collect three independent human response files.
- [ ] Collect and validate registered real camera captures.

## RVTA-Context v1 (2026-08-27)

- [x] Define trusted-reference verification, numeric-capture, and exact-read endpoints.
- [x] Implement license/GPS/time filtering and nearest-station joins against public Singapore temperature data.
- [x] Collect 12-item development and 100-item candidate source sets without victim-conditioned selection.
- [x] Freeze attribution, source/fact hashes, counterfactual severity rules, and rejection logs.
- [x] Implement deterministic perspective-paper rendering and generate the complete 12-item/120-row smoke manifest.
- [x] Add held-out fail-closed review requirements and three randomized 100-item review forms.
- [x] Add Qwen-3B/7B and four-model n=100 configs, runner, summarizer, tests, task packet, and dataset card.
- [ ] Complete three independent source reviews and freeze the approved outdoor subset.
- [ ] Rebuild the 100-item rendered manifest on `/disk2` (local repeated renders exceeded workspace disk budget).
- [ ] Run the four-model pilot/full inference and report real grounded ASR and false-value capture.
- [ ] Add a matched AI-assisted blank-carrier renderer with a fixed generation budget; do not call the deterministic carrier AI-generated.
- [ ] Add at least two independent fact families before presenting RVTA-Context as a general benchmark.

## SCEI-Search adaptive UI (2026-08-27)

- [x] Freeze one immutable false numeric record, uncertainty, question, and source image per run.
- [x] Implement four-state target-flip/exact-read feedback and constrained next-round interventions.
- [x] Enforce a visible 1--12-round black-box budget and clean-correct eligibility gate.
- [x] Preserve all rounds, hashes, masks, query counts, Success@K, and budget-exhaustion outcomes.
- [x] Add a Hugging Face-style Gradio entry point, live trace, and downloadable audit bundle.
- [x] Run compiler, renderer, adaptive-loop, feedback-state, syntax, and bundle smoke checks locally.
- [x] Run a real GPU-backed Gradio session on server 212 and verify the page and `/run` endpoint.
- [x] Freeze and report the development-disjoint 50-item adaptive run without using outcomes to change the selection.
- [x] Add the complete 50-item adaptive aggregate to the paper; Success@1 and Success@2 are both 9/50.
- [x] Complete the four-model SCEI-Images-300 matched flat/scene victim evaluation: four 1,500-row audited logs, path-free aggregate release, strict grounded analysis, controls, Wilson intervals, and exact McNemar tests.
