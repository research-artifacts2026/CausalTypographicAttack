# Frozen media and object-association diagnostic

All 4,608 calls completed without runtime errors or unparsed decisions. The
registration, request manifests, executed code and 640 original PNG assets were
audited before scoring. All 64 object-swap pairs preserve every pixel at and
below row 370. Portable replay:

```bash
python scripts/analyze_channel_study.py --replay evidence/channel_binding_n128
```

This directory includes exact prompts, choice maps, raw responses, timestamps,
input traces, per-item paired vectors, full-set scores, ten prespecified tests,
and source references by cryptographic hash. `study_design.json` normalizes
private checkpoint locations; its original registration hash is retained.
The original source photographs and rendered images remain in the research
archive and are not redistributed in this text-only code release. Obtain
COCO/VOC sources from their original providers under their respective terms.

`runtime_checkpoint_snapshot.json` records the installed package versions and
hashes of checkpoint weights, tokenizers and configuration files. It was
captured after inference and is explicitly retrospective; it must not be
represented as an additional pre-inference weight attestation. The original
prospective registration remains unchanged.

## Media: 128 reused archived scenes, two record states

Full-scene pair accuracy is 3/128 for Qwen2.5-VL-7B and 8/128 for Qwen3-VL-8B.
Adding exact supplied task fields raises it to 24/128 and 35/128: +16.40625 pp
(95% paired bootstrap CI 10.15625 to 22.65625, Holm p=0.000005722) and +21.09375
pp (12.5 to 29.6875, p=0.000017327). Supplied fields change the representation
and may affect attention; the result does not isolate OCR.

With the identical field-supplied prompt but genuinely no image input, pair
accuracy is 12/128 and 19/128. The latter drop of 12.5 pp has Holm p=0.0280015;
the former drop has p=0.0644531. Removing the scene while retaining the record
pixels gives 11/128 and 13/128; neither contrast with the full scene survives
the six-test Holm family. Context removal is not equivalent to image-free input.
Exact independent reads pass both states for 45/128 and 19/128 items. Only
4 and 2 items also pass both text-only decisions; their image-pair failures
are 4 and 1, respectively. These small conjunctions are descriptive only.

## Object association: 64 digital two-panel items

Segmentation-isolated VOC objects of different classes exchange panels. The
two printed records remain fixed, one valid and one invalid; the requested
object determines which to assess. This creates a visual dependency absent
from the self-contained single-record task. Records are hypothetical task
inputs, not measurements of the photographed objects.

Image-only pair accuracy is 11/64 and 5/64. Supplying both exact fields and the
target location gives 12/64 and 13/64. None of the four prespecified contrasts
(fields versus both; location versus both, for each model) survives its Holm
family. Independent localization succeeds on both swaps for 64/64 and 63/64;
single-record image-free reasoning succeeds for only 12/64 and 9/64. Among
items passing both independent prerequisites, 10/12 and 7/8 still fail the
image-only pair. This selected, separate-call conjunction does not isolate
an internal association mechanism or establish general reasoning competence.

## Design and boundaries

All decisions/localizations allow 96 output tokens and request a bare option;
independent transcription allows 384 tokens. Decoding is greedy. Exact fields
and locations are oracle annotations, not outputs of a practical defense.
They change input length; equal calls are not equal input-token budgets.
Both studies use all-item pair accuracy, not the historical three-state DC-ASR.
The primary population is reused archived records, not held-out source transfer.
Bootstrap resampling is paired and stratified by source (media) or template
family (association), 10,000 draws, seed 20260914. Intervals are marginal;
Holm-adjusted tests govern multiplicity. Null tests do not establish equivalence.

The RGB segmentation masks are decoded with the official VOC palette,
excluding background and void. See the original
[VOC segmentation examples](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/segexamples/index.html).
Reconstruction uses `register_channel_study.py` with original confirmation
packets, the VOC train parquet archive and the registered DejaVu font. Install
`pyarrow` for this construction step. `run_channel_study.py` runs each frozen
model/shard once; interrupted unknown calls block silent retries.

`build_blinded_channel_review.py` prepares an outcome-blind 96-case sample,
randomized by fixed hashes, plus an empty CSV and offline review page. Its
answer key is outside the review ZIP. Independent human annotations have
not been supplied and are not claimed as evidence.
