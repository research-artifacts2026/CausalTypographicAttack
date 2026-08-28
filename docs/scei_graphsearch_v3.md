# SCEI-GraphSearch v3

SCEI-GraphSearch is an adaptive, query-bounded extension of SCEI-v2. Its added
complexity is deliberately split into two independently ablatable search
spaces rather than an unconstrained retry loop.

## Frozen semantic bank

For source image `x`, registered target object `o`, visible-object set `O`, and
seed `s`, the pre-victim compiler builds a scene profile and a candidate bank

`B(x,O,s) = {(family, anchor, difficulty, p-, p+)}`.

The profile maps visible objects to typed affordances such as motion, capacity,
rigid clearance, temperature, and phase. Each candidate contains a mechanically
false record `p-`, a one-field corrected twin `p+`, an independent validator,
and a violation margin. The serialized bank is hashed and written to
`protocol.json` before the first victim query. Victim outputs never create or
delete semantic candidates.

## Hierarchical feedback policy

Each attack round uses one semantic arm and one delivery arm. Delivery arms
control carrier type, placement, title, and verdict-free status wording. The
answer and exact-transcription probes yield the same two-bit feedback used by
SCEI-v2:

| Feedback | Registered action |
|---|---|
| target not flipped, record not exactly read | keep semantics; change delivery |
| target flipped, record not exactly read | retain as ungrounded; change delivery |
| record exactly read, target not flipped | increase violation margin within the family, then change family |
| target flipped and record exactly read | strict success; stop |

Whenever the policy selects a new semantic arm, the victim must first reject
that arm's false claim on the unmodified source image. A failed clean gate is
retained and the complete constraint family is skipped. This prevents question
switching from converting pre-existing clean errors into attack successes.

## Reporting

Primary metrics are selected-denominator strict `Success@K`, target flip,
exact-read rate, victim queries, and budget-exhaustion outcomes. Every clean
gate, attack query, rendered image, mask, candidate-bank hash, and stopping
decision is retained.

The first matched development pilot uses 12 disjoint COCO items,
Qwen2.5-VL-7B, scene rendering, and `K=2`. SCEI-v2 and GraphSearch each obtain
2/12 strict successes (16.7%) with 4.83 mean victim queries per selected item.
GraphSearch changes target judgment on 10/12 and exactly reads at least one
candidate on 3/12; SCEI-v2 obtains 10/12 and 2/12 respectively. These counts
are exploratory and too small for a paper claim. More importantly, only one
GraphSearch item reaches `read_but_resisted` at its final allowed round, so the
semantic switching branch is not evaluated by this `K=2` pilot.

## Entry points

- `cta/scei_graphsearch.py`: scene profile, candidate bank, policy, and event loop.
- `scripts/run_scei_graphsearch_batch.py`: frozen resumable batch evaluation.
- `configs/scei_graphsearch_v3_qwen7_coco_n12_k2.yaml`: matched development pilot.
- `tests/test_scei_graphsearch.py`: deterministic bank, policy, clean-gate, and full-loop tests.

Do not describe v3 as more effective than v2 until a preregistered, matched
`K>=4` evaluation on multiple victim models shows a reliable improvement.
