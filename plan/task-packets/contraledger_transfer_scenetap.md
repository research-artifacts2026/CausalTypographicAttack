# ContraLedger transfer and matched-delivery completion

## Objective

Close two declared evidence gaps without changing the attack after observing
victim outputs:

1. freeze and evaluate 200 family-balanced Pascal VOC 2012 images as an
   independently routed source dataset;
2. compare ContraLedger's native carrier with a content-matched naive flat
   renderer and a SceneTAP pipeline replay under the identical three-state
   question, item set, option map, registered text, and success criterion.

## Evidence boundary

- The SceneTAP replay uses the public SoM and TextDiffuser components but a
  local Qwen2.5-VL planner because the original GPT-4o service is unavailable.
- This is a matched delivery-layer comparison. It is not an exact reproduction
  of SceneTAP's original short target-token attack content.
- All items, texts, truth twins, questions, and candidate-index rules are
  frozen before victim inference. No successful-example filtering is allowed.
- Corrected/false twins reuse the same selected region, binary mask, candidate
  index, and diffusion seed; any planner parse/region/caption fallback is fixed
  and counted in provenance before victim inference.
- Main claims require complete logs and fail-closed manifest/hash audits.

## Planned matrix

- Datasets: COCO val2017 (existing frozen n=200), Pascal VOC 2012 segmentation (new
  frozen n=200; 25 items per counterfactual family).
- Delivery: native scene-adaptive carrier, content-matched naive flat carrier,
  public SceneTAP SoM/TextDiffuser with local Qwen placement planner.
- Victims: Qwen2.5-VL-3B, Qwen2.5-VL-7B, LLaVA-OneVision-1.5-8B, InternVL2-8B.
- Endpoint: false-record CONSISTENT target ASR conditioned on correct
  source-ABSENT and true-record-CONSISTENT judgments; EOR additionally requires
  exact false-record transcription and independent rule rejection.

## Completion checks

- source independence, family balance, and hashes audited;
- every dataset/method/model cell complete with all 600 decision rows and all
  required Read/Knowledge probes;
- tables regenerated from evidence JSON, never hand-entered;
- paper wording distinguishes exact public components from official-equivalent
  SceneTAP reproduction;
- LaTeX compiles and paper-state/build hashes are refreshed.
