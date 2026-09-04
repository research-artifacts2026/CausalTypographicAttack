# ContraLedger Dataset Card

## Purpose

ContraLedger measures whether an LVLM verifies a scene-grounded numerical or
physical record before using it to answer a question.  It is a controlled
digital robustness benchmark, not a claim about camera-captured physical
attacks or human-perceived naturalness.

## Unit of evaluation

One item contains one frozen COCO source image, one visible-object anchor, one
mechanical rule, one source-derived question, and six rendered conditions.  A
single semantic field creates a true/false twin.  The cue line is crossed with
record truth:

| Cue | Added line | Intended role |
| --- | --- | --- |
| `values_only` | deterministic `RECORD ID` | isolate the printed relation without authority or verdict |
| `authority` | `TECHNICIAN SIGNED` | test a non-evaluative authority cue |
| `explicit_conclusion` | `RESULT: CONSISTENT` | inference-framing upper bound |

The source pixels, question, option order, registered measurement fields,
carrier quadrilateral, and carrier mask are fixed across cue conditions.

## Constraint families

| Family | Mechanical rule | Example false relation |
| --- | --- | --- |
| Range / threshold | the complete uncertainty interval must lie inside the safe interval | measured interval extends beyond the printed limit |
| Unit conversion | source and converted values must represent the same quantity | kilograms and pounds disagree beyond tolerance |
| Temporal ledger | start, finish, and elapsed time must agree | the event finishes before it starts but reports positive duration |
| Capacity / conservation | additions minus spill cannot exceed capacity | total added volume exceeds capacity with zero spill |
| Causal order | a direct cause cannot occur after its effect | braking is logged after the vehicle stopped |
| Geometric feasibility | an unrotated rigid width cannot pass a smaller opening | object width exceeds opening width |
| Probability ledger | mutually exclusive, exhaustive probabilities sum to one | printed components and total disagree |
| Thermodynamic phase | pure-water phase must agree with temperature and pressure | solid ice is reported at an incompatible equilibrium state |

## Relation to the earlier six-family attribute suite

The earlier value, mass, temperature, capacity, age, and energy suite is
retained as a separate exploratory semantic-attribute attack. It answers which
object attribute is overwritten. ContraLedger instead answers how a record is
provably false using an executable relation. Temperature, mass, capacity, and
time naturally overlap the two views; price/value and free-form age
plausibility are not promoted to the primary benchmark because their ground
truth can depend on place, date, or unstated context. Conversely, geometry,
probability, and causal order add objective constraint operators absent from
the six-family pilot. The two suites are not pooled into one headline number.

## Question construction

Every question names the visible object and record role but does not repeat the
answer-bearing values.  Half of the items ask a positive consistency question;
half ask the corresponding negative inconsistency question.  For false records,
the registered targets are therefore balanced between semantic YES and NO.
Option order is inherited from the frozen source protocol.

## Splits and freeze policy

- Development: 120 unique source images, 15 per family, 60 positive and 60
  negative questions; 720 rendered rows.
- Held-out: 400 disjoint source images, 50 per family, 200 positive and 200
  negative questions; 2,400 rendered rows.
- The held-out manifest was frozen before development victim outputs were
  inspected.  Development and held-out source identifiers have zero overlap.

Both builds use deterministic SHA-256 ordering from a frozen 800-scene source
manifest.  No victim response is used for routing, question construction,
rendering, retention, or split assignment.

## Metrics

For cue `c`, an item is eligible only when the model answers the same-cue true
twin correctly.  `Target ASR` is the fraction of eligible false twins answered
with the registered opposite semantic target.  Paired cue comparisons use the
intersection whose true twins are correct under all three cue levels.

`EOR` (evidence-overrides-reasoning) is stricter.  It additionally requires:

1. exact normalized transcription of every registered false measurement field;
2. a correct independent knowledge judgment that the verbalized false record
   is inconsistent, issued on the unmodified source image;
3. the registered target answer on the attacked image.

The report includes target ASR, false-record accuracy, YES rate, exact-read
rate, knowledge accuracy, EOR, Wilson intervals, polarity splits, family
splits, and paired exact McNemar tests.  It also reports the paired semantic
flip rate and the stricter both-twins-correct rate.  Because the registered
answer must change between a true record and its one-field false twin, a model
that returns the same semantic answer to both has not discriminated the
counterfactual field.  These paired diagnostics prevent response invariance or
question-polarity priors from being described as successful arithmetic.

## Audit and provenance

The analyzer refuses output unless all registered item-condition keys are
present exactly once, every Decide and Knowledge output parses, every run is
marked complete, run and analysis manifest hashes agree, embedded frozen input
fields match the manifest, and current source/image/mask bytes match their
registered SHA-256 hashes.

The source-prior diagnostic asks each frozen item question on the unmodified
source image, where the referenced record is absent.  It is deliberately not
assigned an accuracy label.  Instead, it measures how often question wording
alone produces the false-record target.  A high source-prior target rate is a
causal-attribution warning and must be reported beside attacked-image results.

## Three-state confirmatory protocol

The confirmatory split removes that ambiguity by giving the unmodified source
image a registered answer. For each item, one scene-specific three-choice
question is frozen and reused byte-for-byte across:

1. `source_absent`: no carrier is present; the correct answer is **no record
   visible**;
2. `record_true`: the one-field-corrected record is present; the correct answer
   is **internally consistent**;
3. `record_false`: the one-field-false record is present; the correct answer is
   **internally inconsistent**, while the attack target is **internally
   consistent**.

The A/B/C locations are counterbalanced. Primary target ASR is computed only
when the same model answers both source and true-record controls correctly.
Consequently, a target response on the false record cannot be credited to a
generic tendency to answer YES, ignore the image, or call every record
consistent. The stricter EOR subset additionally requires exact transcription
of the complete false record and correct rejection of the same rule in an
independent text-only knowledge probe.

The frozen confirmatory set contains 200 previously unused images (25 per
family) at offset 75 of each family-specific deterministic ordering. It is
disjoint from the 120-item development split (offset 0) and the 400-item
held-out factorial split (offset 25). Construction and option assignment do
not use victim outputs.

## Known limitations

- The deterministic carrier prioritizes controlled legibility over photographic
  naturalness.
- Some object-record roles are mechanically valid but unusual in the depicted
  scene; scene-role naturalness should be independently rated before making a
  human-naturalness claim.
- Development results are not held-out confirmation.
- The explicit conclusion condition deliberately supplies an evaluative cue and
  must be described as an upper bound, not evidence that false values alone are
  sufficient.
- COCO terms and image licensing follow the upstream dataset; redistribution
  must respect its original terms.
