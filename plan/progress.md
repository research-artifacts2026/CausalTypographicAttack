# RVTA-Bench implementation progress

- [x] Freeze scope, partitions, conditions, labels, and prohibited claims.
- [x] Implement exact-area and benign-evidence controls.
- [x] Implement the disjoint held-out factorial split.
- [x] Implement truth-aware transfer metrics and evidence generation.
- [x] Add model configurations and annotation validation.
- [x] Run tests and manifest smoke checks for the original RVTA pipeline.
- [ ] Run matched and factorial inference.
- [ ] Generate completed evidence assets.
- [ ] Revise and push the paper.

## Public benchmark extension

- [x] Freeze RIO-Bench, SceneTAP, REALM, and PPIA public-code commits and threat-model boundaries.
- [x] Implement RIO Obj-MC materialization, question-conditioned CTA generation, official-score replay, and deterministic simulated capture.
- [x] Register Qwen-3B/7B, LLaVA-OneVision-8B, and InternVL2-8B configs and strict run-completeness audit.
- [ ] Materialize the official RIO val pilot on server 212.
- [ ] Run Qwen-3B/7B pilot and official RIO scorer; expand only after completeness checks.
- [ ] Run LLaVA/InternVL transfer under the identical manifest.
- [ ] Complete full SceneTAP, real physical capture, and three-person independent evaluation.

## RVTA-QA balanced-v1 and AI-natural carrier (2026-08-25)

- [x] Implement truth-polarity, A/B-order, and AB-versus-YES/NO counterbalancing.
- [x] Score correct and target responses semantically per item.
- [x] Preserve fixed paired questions and independent claim-transcription audits.
- [x] Add COCO/VOC configurations for four open LVLM checkpoints.
- [x] Add build, run, validation, CSV/JSON, and LaTeX generation scripts.
- [x] Add six-cell unit tests and run the local smoke suite.
- [x] Generate three provenance-locked AI natural-carrier examples.
- [x] Prepare a blinded three-person naturalness pack with hidden repeats.
- [ ] Materialize the two 300-item manifests on server 212.
- [ ] Run eight complete model-dataset jobs and validate hashes.
- [ ] Run the four-model n=3 natural-carrier pilot (qualitative only).
- [ ] Collect three independent human response files.
- [ ] Collect and validate registered real camera captures.
