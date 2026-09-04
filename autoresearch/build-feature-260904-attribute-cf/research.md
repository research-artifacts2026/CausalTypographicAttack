# Autoresearch log: attribute counterfactual pilot

## Objective

Build, freeze, run, and audit a 120-item six-family minimal-counterfactual pilot on Qwen2.5-VL-7B and LLaVA-OneVision-1.5-8B, then integrate only complete real results into the paper.

## Acceptance predicate

The manifest must be frozen before victim inference; contain 120 unique items with 20 per family, balanced semantic targets and A/B order; implement the full fact-by-conclusion factorial; log independent Read, Ground, Verify, and Decide calls; complete both model runs without missing rows; and pass an independent evidence aggregation before paper claims are changed.

## Iteration 0

- Classification: build-feature.
- Reused the existing deterministic renderer and model adapters.
- Added a dedicated attribute-counterfactual protocol rather than altering frozen RVTA-QA or SCEI protocols.
- Network access to the user-authorized 212 host is currently blocked by the task sandbox; local implementation and tests proceed without inventing results.

## Evidence boundary

No pilot results are available at this point. Existing SCEI results must not be relabeled as this experiment.

## Iteration 1

- Implemented six families with three YES-target and three NO-target directions.
- Restricted source labels so each corrected attribute is ordinarily plausible for its object class.
- Removed question wordings that made the printed false value definitionally true; downstream questions now ask about the visible object under ordinary-world constraints.
- Crossed attribute truth with one fixed target-semantic conclusion, preserving identical conclusion text across the two conclusion-present cells.
- Added exact Read, semantic Ground/Verify/Decide, common-clean eligibility, factorial effects, family breakdown, and KDI aggregation.
- Five local protocol checks pass: family balance, unique scene routing, minimal-twin geometry, low question/overlay overlap, and synthetic aggregation.
- The existing 800-scene source manifest contains enough disjoint compatible objects for exactly 20 items per family.

## Current blocker

The task sandbox denied SSH network access to the user-authorized 212 host, and no app terminal is attached to this thread. Therefore rendered pixels have not been frozen on the server and no Qwen-7B or LLaVA result exists. No fabricated or proxy result is recorded.
