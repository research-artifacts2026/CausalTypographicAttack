# RVTA-Context v1 working release

This folder documents the reproducible build.  Generated sources and renders
live under `runs/` and are intentionally not treated as paper evidence until
review and victim inference are complete.

## Current artifacts

- `runs/rvta_context_sg_sources_n12`: complete 12-item development source set.
- `runs/rvta_context_sg_n12`: complete 120-row rendered development manifest.
- `runs/rvta_context_sg_sources_n100`: complete 100-item candidate source set,
  with 100 unique station-minute facts and attribution.
- `runs/rvta_context_sg_review_n100`: three independently randomized blind
  source-review forms.
- The 100-item rendered directory is not retained locally because repeated
  render copies exceeded the local workspace disk budget.  It is rebuilt from
  the frozen source manifest on `/disk2` after source review.

## Collect

```bash
python scripts/collect_singapore_weather_slice.py \
  --output-root runs/rvta_context_sg_sources_n100 \
  --limit 100 --candidate-limit-per-category 600 \
  --minimum-luminance 72 \
  --category "Category:Singapore photographs taken on 2023-08-20" \
  --category "Category:Singapore photographs taken on 2023-08-22" \
  --category "Category:Singapore photographs taken on 2023-11-11" \
  --category "Category:Singapore photographs taken on 2023-11-12"
```

The collection command uses only license/metadata/brightness/fact-join rules;
it never calls a victim model.

## Review and freeze

```bash
python scripts/make_contextual_review_pack.py \
  --source-manifest runs/rvta_context_sg_sources_n100/sources.jsonl \
  --output-root runs/rvta_context_sg_review_n100 \
  --annotators 3 --seed 20260827

python scripts/analyze_contextual_review.py \
  --source-manifest runs/rvta_context_sg_sources_n100/sources.jsonl \
  --response runs/rvta_context_sg_review_n100/responses/annotator_1.json \
  --response runs/rvta_context_sg_review_n100/responses/annotator_2.json \
  --response runs/rvta_context_sg_review_n100/responses/annotator_3.json \
  --output-manifest runs/rvta_context_sg_sources_n100/approved_sources.jsonl \
  --minimum-annotators 3

python scripts/build_contextual_counterfactual.py \
  --source-manifest runs/rvta_context_sg_sources_n100/approved_sources.jsonl \
  --output-root runs/rvta_context_sg_n100 \
  --stage held-out
```

`--stage held-out` fails closed unless every source has positive outdoor,
location, and carrier-region review fields.

## Four-model pilot and full run

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_contextual_counterfactual.py \
  --config configs/rvta_context_sg_qwen3_n12.yaml

CUDA_VISIBLE_DEVICES=1 python scripts/run_contextual_counterfactual.py \
  --config configs/rvta_context_sg_qwen7_n12.yaml
```

After the 12-item pilot verifies parsing and OCR, use the four `n100` configs.
The cross-model analysis command is:

```bash
python scripts/analyze_contextual_counterfactual.py \
  --model-log Qwen-3B=runs/rvta_context_sg_qwen3_n100/predictions.jsonl \
  --model-log Qwen-7B=runs/rvta_context_sg_qwen7_n100/predictions.jsonl \
  --model-log LLaVA-8B=runs/rvta_context_sg_llava_n100/predictions.jsonl \
  --model-log InternVL-8B=runs/rvta_context_sg_internvl_n100/predictions.jsonl \
  --output-dir runs/rvta_context_sg_analysis_n100
```

## Evidence status

No model ASR is reported yet.  The source and rendering artifacts are data
construction evidence only.  A static image must never be described as a live
temperature sensor, and synthetic carriers must never be described as camera
captures or AI-generated unless that specific renderer was used and logged.
