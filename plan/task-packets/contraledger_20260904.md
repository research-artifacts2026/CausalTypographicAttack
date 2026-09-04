# ContraLedger truth-by-cue experiment

## Frozen question

Do scene-grounded, mechanically false records alter LVLM decisions when the
question, scene, numeric fields, renderer geometry, and answer order are fixed;
and does the effect persist without an explicit consistency verdict?

## Design

- Eight mechanically validated constraint families from the frozen SCEI source pool.
- One scene-specific question per item, byte-identical across conditions.
- Positive and negative consistency questions are balanced within the frozen
  selection so an always-YES response cannot masquerade as attack success.
- A 2 x 3 factorial: record truth (true/false) by cue level (values-only,
  non-evaluative authority, explicit conclusion).
- The true/false twin differs only in the registered corrected field within
  each cue.  Cue levels change only the final record line.
- Development pilot: 15 items per family (120 items); held-out expansion uses
  a disjoint per-family offset and is not selected using pilot victim outputs.

## Endpoints

- Primary: target ASR for each false condition, conditioned on the same-cue
  true twin being answered correctly.
- Mechanism: Epistemic Override Rate (EOR), which additionally requires exact
  transcription of the false record and a correct independent rule judgment.
- Report all models, families, denominators, Wilson intervals, cue-level paired
  gains, and negative results.
- Report paired semantic-flip and both-twins-correct rates. Because the
  registered answer changes across the one-field twin, invariant responses are
  failed discrimination rather than successful arithmetic.
- Run a source-prior diagnostic with the identical frozen question on the
  unmodified image. This has no accuracy label; it measures how often question
  wording alone already produces the false-record target.

## Claim boundary

Values-only isolates numeric/relational content.  Authority is non-evaluative.
Explicit conclusion is an attack-strength upper bound, not evidence that the
model failed arithmetic.  Rendering is deterministic synthetic integration,
not a physical capture or human-naturalness result.

## Stopping rule

Run every frozen row exactly once for independent Read, Knowledge, and Decide
queries.  Do not alter questions, records, items, or cue wording from victim
outputs.  Freeze a disjoint held-out manifest before inspecting any expansion
outcomes.

The source-prior diagnostic is exhaustive over the frozen item set and uses
one query per item and checkpoint. It may weaken, but must never be omitted to
strengthen, the causal interpretation.

## Three-state confirmation

- [x] Freeze a disjoint 200-item split with source-absence, true-record, and
  false-record conditions.
- [x] Counterbalance semantic A/B/C positions and keep one byte-identical
  scene-specific question per triplet.
- [ ] Complete four-model runs. Primary target ASR requires correct source and
  true-record controls; EOR additionally requires exact reading and correct
  independent rejection.
