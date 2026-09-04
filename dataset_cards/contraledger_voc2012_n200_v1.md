# ContraLedger-VOC2012-200

## Purpose

This is the independently sourced transfer split for the ContraLedger
three-state evaluation. It tests whether the eight executable counterfactual
families transfer from COCO val2017 to Pascal VOC 2012 scenes without using
victim-model outputs for selection, generation, or stopping.

## Source and selection

- Source: Pascal VOC 2012 semantic-segmentation images materialized from the
  public `nateraw/pascal-voc-2012` mirror.
- Frozen source pool: 2,000 unique images, seed `20260904`.
- Confirmation set: 200 images, exactly 25 per family.
- Selection: deterministic compatibility routing followed by seeded SHA-256
  order; no victim output is consulted.
- Source-pool manifest SHA-256:
  `fc4e2ce0285090a182049663017d1a41dac1cf086bb74678b844d2d0448b818e`.

## Construction

Each source image yields an unmodified source state, a valid record, and a
one-field false record. The record generator creates 200 unique semantic
signatures across range/threshold, unit conversion, temporal ledger, capacity
conservation, causal order, geometric feasibility, probability ledger, and
thermodynamic phase. A clean-image Qwen2.5-VL-3B planner supplies only scene
description, object-grounded wording, carrier type, and placement. It never
sees victim responses or chooses the numerical truth values.

The native renderer is deterministic digital compositing with matched
false/true carrier geometry and masks. It is not diffusion inpainting or
camera capture. The naive-flat and SceneTAP-component conditions are derived
from the same frozen three-state manifest so their item, question, option map,
record text, and target semantics remain fixed.

## Audited dimensions

- Items: 200.
- Three-state rows and unique images: 600.
- Matched attack/control masks: 400.
- Difficulty: 69 subtle, 66 moderate, 65 strong.
- Planner-valid items: 200; planner fallbacks: 0.
- Unique semantic signatures: 200; duplicates: 0.
- One-field false/true pairs: 200/200.
- Geometry- and mask-matched pairs: 200/200.
- Dataset manifest SHA-256:
  `b8500d500558070ca5e238806635ca49476d5931eba6ceb87240d3323f74e2e4`.
- Frozen three-state manifest SHA-256:
  `d439f3b19dc95878ef49572fd103a36a413f0871b91bd33ef6f9c68519630185`.

The audit must report `implementation_hashes_match=true` and zero errors before
the split is used for evaluation.

## Intended use and limits

Use this split for robustness evaluation on systems and images controlled by
the experimenter. It supports a controlled digital transfer claim only. It
does not establish human-rated naturalness, imperceptibility, camera-captured
physical robustness, or performance on documents and industrial dashboards.
Pascal VOC licensing and attribution requirements remain applicable to the
underlying source images.
