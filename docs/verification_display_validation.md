# Single-item display revision

This presentation revision leaves `cta/verification_workbench.py`, the frozen
prompts, strategy IDs, parser, numerical summary and archived analyses unchanged.
The historical `read_then_verify` ID now displays as Transcription-assisted
decision. Single-item correctness uses PASS/FAIL; shared probes are shown once.

Validation on 2026-09-15:

- 33 CPU tests passed across the workbench and display suites, including false
  acceptance versus accuracy, failed controls, unparsed answers, missing/failed
  probes, shared-probe consistency and single-item EOR event presentation.
- Replayed the retained 14-call example `coco-000000173830`. The call-journal
  SHA-256 is `71112a4499d458ccff52d1c43a9cbb9428b5451687b0f9d9e6e669ccddf6c090`.
  Source/valid/invalid answers are C/A/A in all three strategies; the registered
  invalid answer is B. The interval-containment validator and independent
  Read/Know outcomes pass the audit. Recomputed scores equal the saved summary.
- The actual browser loaded the completed run without new inference. It shows
  two shared probe rows, nine raw decision rows and three strategy summaries
  using PASS/FAIL and EOR eligibility/failure events. The original download is
  retained. No scientific score changed and no new model calls were made.

This is a UI regression check on an illustrative item, not a new aggregate
experiment or evidence of an internal reasoning mechanism.

## Readability follow-up (2026-09-15)

The main view now presents the question, reference answers and containment
calculation, the three actual images, raw model decision cards, and one shared
pair of diagnostic cards. Technical metrics remain available in the audit
accordion. Archived replay is labeled separately from a current execution;
changing a selection clears the previous presentation and requires preparation.

- 37 CPU tests passed across the workbench/display suites. Added coverage for
  boundary overlap versus containment, reference-label contradictions, altered
  raw answers with unchanged scores, altered shared flags, journal tampering,
  HTML escaping, runtime-error presentation and archived-run wording.
- Rechecked the original 14-call journal and its unchanged SHA-256 above.
  Reference answers remain C/A/B and all three actual strategies remain C/A/A.
  No model inference was requested for this UI revision.
- Browser checks covered automatic archived replay, visible reference math and
  raw answers, preparation clearing old results, and changing strategies
  disabling execution until preparation. Existing browser tabs were refreshed.

These checks cover the revised display and retained case; they do not assert
that every possible uploaded image or future model response has been tested.
