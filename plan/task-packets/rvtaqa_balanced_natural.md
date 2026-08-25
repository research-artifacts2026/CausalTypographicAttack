# Task packet: RVTA-QA balanced-v1 and synthetic natural carrier

## Claim being tested

Scene-compatible but reality-violating text can change an otherwise-correct
world judgment after the victim has demonstrably read the claim.  The new
protocol asks whether this effect survives truth-polarity, option-order, and
response-format counterbalancing and a more natural synthetic carrier.

## Frozen primary comparison

- Datasets: COCO (300) and Pascal VOC (300), sampled from the already frozen
  source registries.
- Victims: Qwen2.5-VL-3B/7B, LLaVA-OneVision-1.5-8B, InternVL2-8B.
- Six cells: false/true proposition crossed with AB-no/yes, AB-yes/no, and
  semantic YES/NO response formats. Counts differ by at most one.
- Six image conditions: clean, benign control, direct answer, plain claim,
  Evidence CTA, Causal-Bridge CTA.
- One verification query and one independent transcription query per row;
  greedy decoding and no retries.
- Primary endpoint: semantic grounded clean-conditioned target ASR.
- Every row must match the frozen image hash; incomplete logs are not pooled.

## Synthetic natural-render pilot

Three AI-edited carrier examples preserve the source scene while adding
perspective, texture, environmental light, and shadows. They are labeled
`synthetic natural-render; not real physical capture`. The registry retains
the exact source/output hashes and prompts. The n=3 pilot is qualitative and
cannot supply a headline or physical-world claim.

## Stop conditions

- Do not call the result SOTA unless it wins on a public protocol with matched
  attacks, victims, denominators, and official scoring.
- Do not report a balanced-v1 table from partial model logs.
- Do not relabel model ratings as human ratings.
- Do not report AI edits or simulated camera degradation as physical capture.
- Retain zeroes as observed zero successes; never replace them with missing or
  favorable values.

## Current state

- Code, tests, four-model configurations, natural assets, provenance registry,
  and the three-person blind pack are complete locally.
- Cross-model inference is pending because server 212 is unreachable from the
  current session.
- Three independent human response files and real camera photographs remain
  uncollected.

