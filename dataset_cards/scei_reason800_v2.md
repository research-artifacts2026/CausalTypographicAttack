# SCEI-Reason-800

## Purpose

SCEI-Reason-800 is a controlled image-first benchmark for testing whether a
vision-language model accepts a readable but mechanically inconsistent record.
It is a robustness research artifact, not evidence that any model was attacked
successfully.

## Size and structure

- 800 unique COCO val2017 source scenes.
- Eight counterfactual reasoning families, exactly 100 source scenes each.
- Three matched variants per source: clean, false record, corrected twin.
- 2,400 rendered images and 1,600 registered carrier masks.
- Per-family split: 70 train, 15 validation, 15 test.
- Aggregate split: 560 train, 120 validation, 120 test source items.

## Registered families

1. `range_threshold`: a measured interval lies outside a registered safe range.
2. `unit_conversion`: inconsistent Celsius/Fahrenheit, km/mile, kg/lb, or litre/US-gallon fields.
3. `temporal_ledger`: start, finish, and elapsed time do not agree.
4. `capacity_conservation`: additions, capacity, and recorded spill violate volume conservation.
5. `causal_order`: the claimed direct cause is timestamped after the outcome.
6. `geometry_feasibility`: a rigid unrotated object is wider than its opening.
7. `probability_ledger`: mutually exclusive exhaustive probabilities do not sum to one.
8. `phase_state`: the stated water phase conflicts with temperature at 1 atm.

## Truth controls

Each false record has a one-field corrected twin with identical source image,
carrier geometry, placement, title, and non-changing fields.  Every record
stores its generator version, difficulty, changed field, solver expression,
numeric parameters, residual, and tolerance.  The audit script reparses the
printed fields and recomputes both residuals independently.

## Selection and leakage boundary

Source selection, family allocation, symbolic values, and train/validation/test
splits are frozen before planner or victim inference.  The scene planner sees
the clean image, visible labels, and family name; it never receives victim
outputs and never selects the numeric truth values.  Test items must not be
used for prompt, rendering-policy, or attack-text selection.

## Rendering boundary

Carriers use deterministic perspective, local tone, texture, placement, and
shadow compositing.  They are synthetic and are not camera captures, diffusion
inpainting, or evidence of human-perceived naturalness.

## Required evaluation

Report model-specific clean-correct denominators, target-judgment rate, exact
transcription rate, strict grounded ASR, family-level intervals, and all zero or
failed cases.  Dataset rows alone must never be reported as attack successes.
