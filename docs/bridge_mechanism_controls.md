# Causal-Bridge mechanism controls

This pipeline implements the preregistered Experiment A that separates a false proposition from a target-semantic conclusion. It creates no victim result during dataset construction and must not be tuned after target-model inference.

## Conditions

Every item has a clean image and six attacked conditions with the same title, renderer, box, placement, area, seven nonempty body lines, and near-matched word count:

- `plain`
- `target_only`
- `neutral_only`
- `bridge_aligned`
- `bridge_neutral`
- `bridge_reversed`

The independent read query must recover every registered body field. A partial read, parse failure, clean error, or non-target answer is not a grounded attack success.

## 1. Build and freeze manifests

Run separately for the already frozen balanced COCO and Pascal VOC item registries:

```bash
python scripts/build_bridge_mechanism_controls.py \
  --balanced-items runs/rvtaqa_balanced_coco_n300/items.json \
  --output-root runs/bridge_mechanism_coco_n300 \
  --dataset COCO --expected-items 300 --stage held-out

python scripts/build_bridge_mechanism_controls.py \
  --balanced-items runs/rvtaqa_balanced_voc_n300/items.json \
  --output-root runs/bridge_mechanism_voc_n300 \
  --dataset VOC --expected-items 300 --stage transfer
```

The builder writes `render_manifest.jsonl`, `build_provenance.json`, and `freeze_record.json`. Copy the emitted `manifest_sha256` into each model config. Do not change the mechanism module or images after this point; the runner refuses a stale hash.

## 2. Freeze one config per model and dataset

```yaml
protocol_schema_version: cta/bridge-mechanism-controls-v1
source_manifest: runs/bridge_mechanism_coco_n300/render_manifest.jsonl
expected_manifest_sha256: REPLACE_WITH_EMITTED_64_HEX_HASH
expected_items: 300
output_root: runs/bridge_mechanism_coco_qwen3_n300
seed: 20260901
model:
  adapter: qwen25vl
  name_or_path: /absolute/path/to/frozen/model/snapshot
  device: cuda:0
  dtype: bfloat16
  min_pixels: 200704
  max_pixels: 602112
  max_new_tokens: 192
  do_sample: false
```

Use the repository's existing model-adapter settings for Qwen2.5-VL-3B/7B, LLaVA-OneVision-1.5-8B, and InternVL2-8B. The only allowed changes across configs are model adapter/snapshot/device, dataset manifest/hash, and output directory.

## 3. Run victims

```bash
python scripts/run_bridge_mechanism_controls.py --config configs/bridge_mechanism_coco_qwen3_n300.yaml
```

The run is append-only and resumable. `failures.jsonl` records failed inference attempts; a failed row is never silently retried as a successful replacement. Each item uses one verification query and one independent full-field transcription query per condition.

## 4. Analyze only after all eight cells complete

```bash
python scripts/analyze_bridge_mechanism_controls.py \
  --run Qwen3@COCO=runs/bridge_mechanism_coco_qwen3_n300/predictions.jsonl \
  --run Qwen7@COCO=runs/bridge_mechanism_coco_qwen7_n300/predictions.jsonl \
  --run LLaVA@COCO=runs/bridge_mechanism_coco_llava_n300/predictions.jsonl \
  --run InternVL@COCO=runs/bridge_mechanism_coco_internvl_n300/predictions.jsonl \
  --run Qwen3@VOC=runs/bridge_mechanism_voc_qwen3_n300/predictions.jsonl \
  --run Qwen7@VOC=runs/bridge_mechanism_voc_qwen7_n300/predictions.jsonl \
  --run LLaVA@VOC=runs/bridge_mechanism_voc_llava_n300/predictions.jsonl \
  --run InternVL@VOC=runs/bridge_mechanism_voc_internvl_n300/predictions.jsonl \
  --output-dir runs/bridge_mechanism_analysis_n600 \
  --bootstrap-draws 10000 --seed 20260901 --expected-cells 8
```

The primary interaction is

```text
(bridge_aligned - bridge_neutral) - (target_only - neutral_only)
```

The analyzer reports paired source-item bootstrap intervals, a source-clustered binary interaction estimate, secondary aligned-versus-reversed and aligned-versus-target-only contrasts, and the frozen claim gate. If the claim gate is not supported, the paper must not attribute the effect to the false proposition or call the method a causal mechanism.

## Verification

```bash
python -m pytest -q tests/test_bridge_mechanism_controls.py
python -m py_compile \
  cta/bridge_mechanism_controls.py \
  scripts/build_bridge_mechanism_controls.py \
  scripts/run_bridge_mechanism_controls.py \
  scripts/analyze_bridge_mechanism_controls.py
```

This pipeline tests proposition-by-conclusion interaction only. Scene dependence, public-protocol comparison, human validity, natural rendering, and physical robustness require separate registered experiments.
