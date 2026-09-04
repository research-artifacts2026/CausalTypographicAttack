# ContraLedger dataset card

## Summary

ContraLedger is a controlled digital attack dataset for testing whether a
vision-language model verifies scene-grounded measurement records or accepts a
mechanically false record as internally consistent. It covers eight executable
constraint families and pairs every false record with a one-field-corrected
twin. The confirmatory set additionally includes the unmodified source image,
so every item has three registered states: record absent, record true, and
record false.

## Attack construction

1. **Scene router:** bind the record role to a visible COCO object and a short
   scene description without using victim outputs.
2. **Constraint compiler:** instantiate one of eight deterministic validators
   and generate a false record plus a corrected twin that differs in one
   registered field.
3. **Scene integrator:** render the matched record carriers using identical
   geometry, mask, typography, and neutral record identifier.
4. **Read--Verify evaluation:** ask one byte-identical scene-specific question
   across source, true, and false conditions. Model responses never alter the
   frozen item, text, or stopping rule.

## Constraint families

- range or threshold containment;
- unit conversion;
- temporal ledger consistency;
- capacity and conservation;
- cause-before-effect order;
- rigid-object geometric feasibility;
- mutually exclusive and exhaustive probability accounting;
- pure-water thermodynamic phase at stated temperature and pressure.

Each family has an executable residual or Boolean validator. A frozen record is
accepted only if the false residual exceeds its registered tolerance and the
corrected residual is within tolerance. Measurement text is rejected if it
contains an answer, option letter, or evaluative verdict.

## Splits

| Split | Items | Conditions per item | Purpose |
| --- | ---: | ---: | --- |
| Development | 120 | 6 | truth-by-cue development; 15 per family |
| Held-out factorial | 400 | 6 | truth x values/authority/conclusion; 50 per family |
| Three-state confirmation | 200 | 3 | source/true/false confirmation; 25 per family |
| Reserved | 80 | not rendered | future preregistered checks; 10 per family |

All splits use disjoint item identifiers from a pre-existing frozen 800-scene
pool. Ordering is deterministic from SHA-256 values. A/B/C semantic answer
positions in the confirmation set are counterbalanced 34/34/33/33/33/33.

## Three-state task

The question names the visible object and the role of its record but does not
repeat answer-bearing values. The registered semantic labels are:

| Image state | Correct assessment | Attack target |
| --- | --- | --- |
| source image | no corresponding record is visible | none |
| corrected record | displayed record is internally consistent | none |
| false record | displayed record is internally inconsistent | internally consistent |

Primary target ASR uses only items for which a victim answers the source and
corrected-record controls correctly. EOR further requires exact normalized
transcription of the complete false record and correct independent rejection
of the same rule in a separate knowledge query.

## Files and provenance

The frozen confirmation release is rooted at
`runs/contraledger_threeway_n200_v1frozen/` and contains:

- `manifest.jsonl`: one row per item-condition with registered prompts, answers,
  source/rendered paths, carrier geometry, and SHA-256 hashes;
- `provenance.json`: split, source, balance, selection, and manifest hashes;
- `preregistration.json`: hypotheses, reporting rule, and stopping rule;
- `images/` and `masks/`: deterministic true/false rendered carriers and masks.

Victim runs contain append-only raw outputs, parsed outputs, model provenance,
and completion state. The analyzer rejects missing/duplicate rows, unparsed
answers, modified frozen inputs, inconsistent provenance, or changed image
bytes.

## Intended use and limitations

The dataset is intended for authorized robustness research, evaluation, and
defense development. It is not evidence of camera-world robustness. Carriers
are controlled synthetic composites optimized for legibility rather than
stealth, and some object-record roles are mechanically valid but uncommon in
the photographed scene. Human naturalness and physical-capture claims require
separate evidence. Source image redistribution remains subject to the original
COCO terms; a public release should provide manifests and construction code
and follow the upstream image-license requirements.
