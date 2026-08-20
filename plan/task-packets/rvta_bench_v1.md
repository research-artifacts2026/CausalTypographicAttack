# Task packet: RVTA-Bench v1

## Objective

Turn the frozen CTA-v2 study into a matched benchmark that distinguishes attack semantics, evidence presentation, overlay area, and defense utility without test-time policy selection.

## Evidence gates

- [x] Existing discovery and confirmatory test identifiers remain disjoint.
- [x] Register a third held-out factorial split before inference.
- [x] Add same-image naive, scene-aware, original CTA, Evidence CTA, exact-area direct, benign-true, and clean conditions.
- [x] Store expected truth separately from attack name and compute false-ASR versus true-claim utility correctly.
- [x] Add a strict dual-axis human annotation schema and independent-coverage validator.
- [ ] Complete four matched open-model logs.
- [ ] Complete two held-out factorial logs.
- [ ] Materialize licensed BDD100K and WHOOPS! source registries.
- [ ] Collect three independent human annotations per selected item.
- [ ] Regenerate paper assets and revise claims only after complete-log checks pass.

## Stop rules

- Never use the factorial split to change the frozen Evidence CTA policy.
- Never compare rates from different image identifiers as a paired improvement.
- Never treat benign true acceptance as attack success.
- Never publish a model judge as independent human annotation.
