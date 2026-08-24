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
- [ ] Materialize the official RIO val pilot on server 212.
- [ ] Run Qwen-3B/7B pilot and official RIO scorer; expand only after completeness checks.
- [ ] Run LLaVA/InternVL transfer under the identical manifest.
- [ ] Complete full SceneTAP, real physical capture, and three-person independent evaluation.
