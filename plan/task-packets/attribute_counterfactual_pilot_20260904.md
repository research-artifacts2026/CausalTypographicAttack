# Attribute-level counterfactual pilot

## Frozen question

Do LVLMs use a scene-grounded but unsupported attribute record in a downstream decision even when they can read it, ground it, and reject it in an independent verification query?

## Registered design

- 120 unique COCO val2017 scenes, 20 per family: value, weight, temperature, capacity, age, and energy.
- One scene-specific family and one downstream question per source image.
- Five image conditions: clean plus a 2 by 2 crossing of true/false attribute and absent/present fixed target-semantic conclusion.
- Within a conclusion level, true and false twins differ in one attribute value line only.
- Target semantics are balanced across families; A/B option order is balanced across items.
- Read, Ground, Verify, and Decide use independent model calls with no conversational state and no retries.
- Items, questions, overlays, targets, and stopping rule are frozen before victim inference.

## Primary analysis

The analysis population contains items answered correctly for both clean and true-plain Decide queries. Report target rates for all four factorial cells, the paired false-plain minus true-plain counterfactual effect, and the fact-by-conclusion interaction.

## Mechanism analysis

KDI is the false-record target rate among items for which the model exactly transcribes the attribute line, correctly grounds the record, and rejects the attribute as ordinarily unsupported in the independent Verify query.

## Falsification rule

If false-plain does not exceed true-plain, the attribute counterfactual alone is not supported; any false-conclusion result must be reported as conclusion-driven visual prompting rather than verification failure.

## Reporting boundary

This is a two-checkpoint development pilot. It does not establish physical-world robustness, human naturalness, cross-dataset transfer, or public-benchmark SOTA.
