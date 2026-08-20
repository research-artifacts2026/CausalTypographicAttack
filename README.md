# Causal Typographic Attack / Reality Violation Attack

This repository implements a fully logged pilot and 300-image expansion for testing whether an LVLM rejects text that is visually compatible with an image but violates ordinary real-world constraints. It also contains cross-architecture replay, a second-dataset evaluation, a matched PIL-versus-TextDiffuser rendering study, and a blinded human-rating package. It does **not** claim to reproduce SAGE or the full SceneTAP pipeline. Results are generated only from JSONL model logs.

## Scope and task

Each COCO128 image uses the largest normalized detection box as its deterministic target object. The LVLM must return the prominent object, transcribe an added claim, and judge it as `TRUE`, `FALSE`, or `NONE`. Strict attack success requires both (i) accepting the false claim as `TRUE` and (ii) at least 50% content-token recall between the transcription and rendered overlay. The match rule prevents hallucinated unrelated text on a masked image from being counted as a successful attack. Object accuracy is reported separately; it is not a standard COCO detection metric.

Compared conditions:

- `none`: clean image.
- `naive`: high-contrast wrong-class banner.
- `scene_coherent`: plaque-style wrong-class exhibit label; this is an in-house scene-aware baseline, not SceneTAP.
- `causal`: fluent claim naming the visible class while violating physics, biology, decay, or energy constraints.
- `consistency`: lightweight SAGE-style lexical scene-text wrapper. It masks a wrong-class overlay but intentionally passes a causal claim that names the visible object.
- `rapidocr_mask`: deployable RapidOCR 3.9.2 detections above 0.5 confidence, expanded by two pixels and gray-masked without renderer coordinates.
- `ocr_mask`: text-region masking using the renderer's known bounding box. This is an oracle localization upper bound.

## Server-tested environment

- Ubuntu server `NUS-RobustAI`
- 8 x NVIDIA RTX A6000
- Qwen2.5-VL-3B/7B-Instruct, LLaVA-OneVision-1.5-8B-Instruct, and InternVL2-8B local Hugging Face snapshots
- COCO 2017 validation and Pascal VOC 2012 validation
- Python 3.10 and PyTorch 2.5.1+cu121

The tested Python executable is `/disk2/fangxinyue/.venv/bin/python`. Qwen uses the main environment (Transformers 5.9.0). LLaVA is loaded with the isolated compatibility overlay `/disk2/fangxinyue/cta_crossvl_env` (Transformers 4.57.1), and InternVL uses `/disk2/fangxinyue/cta_internvl_env` (Transformers 4.37.2). These overlays keep the common PyTorch/CUDA runtime while pinning model-specific tokenizer and processor dependencies. To install independently, create separate Python 3.10 environments and install the public model repositories' pinned dependencies. Model weights are not included.

## Run

```bash
cd /disk2/fangxinyue/causal_typographic_attack
export CUDA_VISIBLE_DEVICES=0
/disk2/fangxinyue/.venv/bin/python scripts/download_data.py
/disk2/fangxinyue/.venv/bin/python -m pytest tests -q
/disk2/fangxinyue/.venv/bin/python run_experiment.py --config configs/smoke.yaml
/disk2/fangxinyue/.venv/bin/python run_experiment.py --config configs/pilot_qwen25vl3b.yaml
```

## Recommended ablations and extensions

Once baseline and main figures are in place, run these controlled ablations:

```bash
# Overlay-match threshold ablation (replace 0.5 with 0.6)
CUDA_VISIBLE_DEVICES=0 /disk2/fangxinyue/.venv/bin/python run_experiment.py --config configs/pilot_qwen25vl3b_ratio60.yaml

# Cross-fold COCO-2017-HF split ablation (same model, same sample count, disjoint images)
CUDA_VISIBLE_DEVICES=0 /disk2/fangxinyue/.venv/bin/python run_experiment.py --config configs/main_qwen25vl3b_n300_primary.yaml
CUDA_VISIBLE_DEVICES=0 /disk2/fangxinyue/.venv/bin/python run_experiment.py --config configs/main_qwen25vl3b_n300_secondary.yaml

# Build an apples-to-apples 2x300 comparison table after both folds finish
/disk2/fangxinyue/.venv/bin/python scripts/build_paper_table.py runs/main_qwen25vl3b_n300_primary --copy-to paper/generated_primary_results.tex
/disk2/fangxinyue/.venv/bin/python scripts/build_paper_table.py runs/main_qwen25vl3b_n300_secondary --copy-to paper/generated_secondary_results.tex
```

`overlay_match_ratio` is now a config-level ablation control (`0.5` by default).
`dataset.split` (`primary` / `secondary`) creates two disjoint COCO-2017-HF splits via deterministic even/odd partitioning.

The runner is resumable at condition granularity. Re-running the same command skips completed `(sample, attack, defense)` keys.

After validating the 100-sample pilot, run the non-duplicated 300-image COCO val2017 configuration:

```bash
CUDA_VISIBLE_DEVICES=0 /disk2/fangxinyue/.venv/bin/python run_experiment.py --config configs/main_qwen25vl3b_n300.yaml
```

Pre-download with `scripts/download_data.py --dataset coco_val2017_hf`, or let the first experiment call download the two validation Parquet shards from the public `BrandonLSX/coco-2017` Hugging Face mirror. The loader materializes only selected images and reads the included COCO instance annotations. Increase `num_samples` to 500 with a new `output_root` for a larger run; do not repeat COCO128 images to inflate sample count.

## Outputs

Every run directory contains:

- `sample_manifest.json`: selected samples, labels, absolute source paths, and SHA-256 hashes.
- `scene_graphs/*.json`: raw and parsed LVLM scene extraction.
- `quality/*.json`: raw and parsed 1--5 LVLM judge ratings.
- `images/`: rendered and defended images.
- `predictions.jsonl`: append-only per-condition records including prompts' outputs, attack/defense metadata, latency, and timestamps.
- `provenance.json`: config hash, code commit (when available), runtime/model information, metric definitions, start/end time.
- `summary.json`, `summary.csv`, `results_table.tex`: deterministic aggregates generated from `predictions.jsonl`.

## Cross-model and second-dataset evaluation

The transfer runner consumes an already rendered source manifest. This keeps image pixels, overlay text, prompt, labels, and sample identifiers fixed across models; only the LVLM adapter changes. Pascal VOC targets are derived from the largest foreground semantic-segmentation region using the official VOC color palette rather than from model predictions.

```bash
# COCO, cross architecture
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/disk2/fangxinyue/cta_crossvl_env/lib/python3.10/site-packages \
  /disk2/fangxinyue/.venv/bin/python scripts/run_transfer_eval.py \
  --config configs/transfer_llavaov15_8b_n300.yaml
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/disk2/fangxinyue/cta_internvl_env/lib/python3.10/site-packages \
  /disk2/fangxinyue/.venv/bin/python scripts/run_transfer_eval.py \
  --config configs/transfer_internvl2_8b_n300.yaml

# Pascal VOC 2012, matched 300-image construction
CUDA_VISIBLE_DEVICES=0 /disk2/fangxinyue/.venv/bin/python run_experiment.py \
  --config configs/voc2012_qwen25vl3b_n300.yaml
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/disk2/fangxinyue/cta_crossvl_env/lib/python3.10/site-packages \
  /disk2/fangxinyue/.venv/bin/python scripts/run_transfer_eval.py \
  --config configs/transfer_llavaov15_voc2012_n300.yaml
CUDA_VISIBLE_DEVICES=6 /disk2/fangxinyue/.venv/bin/python scripts/run_transfer_eval.py \
  --config configs/transfer_qwen25vl7b_voc2012_n300.yaml
```

`CUDA_VISIBLE_DEVICES` remaps the selected physical GPU to local `cuda:0`; the corresponding YAML files therefore intentionally use `device: cuda:0`.

## SceneTAP component / natural-render comparison

The public SceneTAP repository is installed separately at `/disk2/fangxinyue/scenetap`. Its full multimodal planner was not used because the configured external endpoint rejected image inputs. The registered renderer experiment instead uses SceneTAP's public TextDiffuser component with fixed candidate index 0 and no manual example selection. This is reported as a **SceneTAP TextDiffuser component** or **natural-render proxy**, never as a reproduction of full SceneTAP.

Long causal captions were visibly truncated by this component, so the registered matched comparison uses compact, semantically equivalent claims for both PIL and TextDiffuser. The same 100 sample identifiers and strings are used in both arms.

```bash
/disk2/fangxinyue/.venv/bin/python scripts/prepare_natural_render_source.py \
  --source-log runs/main_qwen25vl3b_n300/predictions.jsonl \
  --limit 100 --output-root runs/natural_render_source_n100

# Render with the public TextDiffuser component.
PYTHONPATH=/disk2/fangxinyue/scenetap_runtime:/disk2/fangxinyue/scenetap \
  /disk2/fangxinyue/.venv/bin/python scripts/render_scenetap_textdiffuser.py \
  --source-log runs/natural_render_source_n100/render_manifest.jsonl \
  --output-root runs/scenetap_textdiffuser_n100 \
  --scenetap-root /disk2/fangxinyue/scenetap \
  --source-attack causal_compact \
  --output-attack causal_compact_textdiffuser \
  --candidate-index 0

# Evaluate the two matched renderers.
CUDA_VISIBLE_DEVICES=0 /disk2/fangxinyue/.venv/bin/python run_experiment.py \
  --config configs/compact_pil_qwen25vl3b_n100.yaml
CUDA_VISIBLE_DEVICES=0 /disk2/fangxinyue/.venv/bin/python scripts/run_transfer_eval.py \
  --config configs/textdiffuser_qwen25vl3b_n100.yaml
```

The completed 100-image Qwen2.5-VL-3B comparison gives 61.00% strict ASR for compact PIL and 74.00% for the TextDiffuser component. These values are from the completed logs and should be interpreted jointly with legibility/integration ratings; they do not establish physical-world robustness.

## Independent human evaluation

The blind package contains 100 matched images for four methods (naive, scene-coherent, compact PIL CTA, and TextDiffuser CTA). Each of three independent annotators receives all 400 randomized items plus a 10% duplicate set for within-rater reliability. Method names are hidden. Annotators score legibility, visual integration, scene fit, and claim impossibility on 1--5 scales using offline HTML forms, then export CSV files.

```bash
/disk2/fangxinyue/.venv/bin/python scripts/make_human_eval_pack.py \
  --pil-log runs/main_qwen25vl3b_n300/predictions.jsonl \
  --compact-pil-log runs/compact_pil_qwen25vl3b_n100/predictions.jsonl \
  --textdiffuser-log runs/scenetap_textdiffuser_qwen25vl3b_n100/predictions.jsonl \
  --output-root runs/human_eval_blind_n100

# After three completed independent CSV files are placed in responses/:
/disk2/fangxinyue/.venv/bin/python scripts/analyze_human_eval.py \
  --pack-root runs/human_eval_blind_n100 \
  --minimum-annotators 3
```

The analyzer refuses to emit aggregate ratings when fewer than three complete independent response files are present. It removes hidden duplicates only after computing within-rater consistency, reports interval-scale Krippendorff's alpha, and uses 10,000 percentile bootstrap draws clustered by matched image identifier for method means and every paired method difference. No human scores are currently claimed in the paper; until all three response files exist, a human-results table must remain absent.

## Extended paper assets

After every registered log is complete, generate the cross-model/dataset table, paired renderer comparison, natural-render qualitative grid, confidence intervals, and machine-readable evidence in one validated step:

```bash
/disk2/fangxinyue/.venv/bin/python scripts/make_extended_paper_assets.py \
  --qwen-coco runs/main_qwen25vl3b_n300/predictions.jsonl \
  --qwen7-coco runs/transfer_qwen25vl7b_n300/predictions.jsonl \
  --llava-coco runs/transfer_llavaov15_8b_n300/predictions.jsonl \
  --intern-coco runs/transfer_internvl2_8b_n300/predictions.jsonl \
  --qwen-voc runs/voc2012_qwen25vl3b_n300/predictions.jsonl \
  --qwen7-voc runs/transfer_qwen25vl7b_voc2012_n300/predictions.jsonl \
  --llava-voc runs/transfer_llavaov15_voc2012_n300/predictions.jsonl \
  --compact-pil runs/compact_pil_qwen25vl3b_n100/predictions.jsonl \
  --textdiffuser runs/scenetap_textdiffuser_qwen25vl3b_n100/predictions.jsonl \
  --output-dir paper_extended
```

The generator validates full sample/method coverage and exits instead of creating partial paper tables.

## Evidence-augmented CTA v2

The v2 attack treats causal typography as a registered search problem instead of choosing one plaque by intuition. It combines three false-claim phrasings, four graphical evidence-card styles, two scale levels, and deterministic lowest-variance corner placement. Discovery and test identifiers use fixed SHA-256 ordering within each violation family followed by deterministic family round-robin, preventing a small discovery set from collapsing to one template family. The selector freezes one global policy using discovery logs only; it cannot inspect a held-out test response.

```bash
# Render 24 v2 candidates plus the original CTA baseline on 20 stratified discovery images.
/disk2/fangxinyue/.venv/bin/python scripts/build_strong_attack_candidates.py \
  --source-manifest runs/main_qwen25vl3b_n300/sample_manifest.json \
  --output-root runs/cta_v2_discovery_stratified_n20 \
  --split discovery --seed 20260820 --discovery-samples 20 --test-samples 100

# Evaluate identical rendered candidates on two stronger checkpoints.
CUDA_VISIBLE_DEVICES=7 /disk2/fangxinyue/.venv/bin/python scripts/run_transfer_eval.py \
  --config configs/cta_v2_discovery_qwen7_n20.yaml
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/disk2/fangxinyue/cta_crossvl_env/lib/python3.10/site-packages \
  /disk2/fangxinyue/.venv/bin/python scripts/run_transfer_eval.py \
  --config configs/cta_v2_discovery_llava_n20.yaml

# Refuse partial logs, then freeze one policy under the registered score.
/disk2/fangxinyue/.venv/bin/python scripts/select_strong_attack_policy.py \
  --candidate-manifest runs/cta_v2_discovery_stratified_n20/render_manifest.jsonl \
  --eval-log runs/cta_v2_discovery_qwen7_n20/predictions.jsonl \
  --eval-log runs/cta_v2_discovery_llava_n20/predictions.jsonl \
  --output runs/cta_v2_discovery_stratified_n20/selected_policy.json
```

The locked selection score is mean strict ASR across discovery models plus `0.10 * grounded transcription - 0.05 * overlay area`; policies below 75% grounded transcription are ineligible. The original CTA is evaluated but cannot win the v2 search. Every failed candidate remains in the discovery logs. A held-out test must be rendered with `--split test --policy-file ...`; do not report discovery ASR as a test result.

After selection, render the disjoint 100-image frozen-policy test and evaluate its two conditions (original CTA and frozen v2) on four checkpoints:

```bash
/disk2/fangxinyue/.venv/bin/python scripts/build_strong_attack_candidates.py \
  --source-manifest runs/main_qwen25vl3b_n300/sample_manifest.json \
  --output-root runs/cta_v2_test_n100 --split test --seed 20260820 \
  --discovery-samples 20 --test-samples 100 \
  --policy-file runs/cta_v2_discovery_stratified_n20/selected_policy.json

# Use the model-specific PYTHONPATH overlays documented above for LLaVA and InternVL.
/disk2/fangxinyue/.venv/bin/python scripts/run_transfer_eval.py --config configs/cta_v2_test_qwen3_n100.yaml
/disk2/fangxinyue/.venv/bin/python scripts/run_transfer_eval.py --config configs/cta_v2_test_qwen7_n100.yaml
/disk2/fangxinyue/.venv/bin/python scripts/run_transfer_eval.py --config configs/cta_v2_test_llava_n100.yaml
/disk2/fangxinyue/.venv/bin/python scripts/run_transfer_eval.py --config configs/cta_v2_test_internvl_n100.yaml
```

Regenerate a table without model inference:

```bash
/disk2/fangxinyue/.venv/bin/python scripts/build_paper_table.py runs/pilot_qwen25vl3b --copy-to paper/generated_results_table.tex
```

Regenerate the paper's qualitative grid, bootstrap statistics, result chart, and semantic-diagnostic table from the completed sample-level logs:

```bash
/disk2/fangxinyue/.venv/bin/python scripts/make_paper_assets.py \
  --main-log runs/main_qwen25vl3b_n300/predictions.jsonl \
  --pilot-log runs/pilot_qwen25vl3b/predictions.jsonl \
  --pilot-image-root runs/pilot_qwen25vl3b/images \
  --transfer-log runs/transfer_qwen25vl7b_n300/predictions.jsonl \
  --ocr-log runs/rapidocr_defense_qwen25vl3b_n300/predictions.jsonl \
  --ocr-conditions runs/rapidocr_masks_n300/conditions.jsonl \
  --paper-dir paper
```

The script checks that all main cells contain the same 300 sample IDs, uses 10,000 paired bootstrap resamples with seed 2026, and refuses to create qualitative grids if their log-derived outcome no longer matches the caption. It also writes mechanism, violation-family, exact-text-repetition, checkpoint-scale-transfer, and practical-defense evidence to `paper/evidence/expanded_analysis.json`; every generated table and result figure is derived from those sample-level logs.

Replay the completed images on the larger Qwen checkpoint without regenerating attacks:

```bash
CUDA_VISIBLE_DEVICES=7 /disk2/fangxinyue/.venv/bin/python \
  scripts/run_transfer_eval.py --config configs/transfer_qwen25vl7b_n300.yaml
```

Build deployable OCR masks, then evaluate the two masked attack conditions with the 3B checkpoint:

```bash
PYTHONPATH=work/rapidocr_deps /disk2/fangxinyue/.venv/bin/python \
  scripts/build_rapidocr_masks.py --config configs/rapidocr_masks_n300.yaml
CUDA_VISIBLE_DEVICES=0 /disk2/fangxinyue/.venv/bin/python \
  scripts/run_transfer_eval.py --config configs/rapidocr_qwen25vl3b_n300.yaml
```

`work/rapidocr_deps` is an isolated, untracked install target. A fresh environment can instead install `rapidocr==3.9.2` and `onnxruntime` normally. `runs/rapidocr_masks_n300/detections.jsonl` stores recognized strings, boxes, confidences, overlay-token recall, and masked-area upper bounds for every image.

Check within-run uniqueness and cross-run image overlap:

```bash
/disk2/fangxinyue/.venv/bin/python scripts/check_manifests.py runs/pilot_qwen25vl3b/sample_manifest.json runs/main_qwen25vl3b_n300/sample_manifest.json
```

## Verified 100-sample pilot

The completed Qwen pilot contains 800 prediction rows (100 samples times eight conditions). Strict false-claim acceptance ASR is 58% for causal typography, 19% for naive typography, and 25% for the plaque-style scene-coherent baseline. The consistency wrapper reduces naive ASR to 0% but leaves causal ASR at 58%; renderer-bbox masking reduces both to 0%. Clean object accuracy under the largest-area proxy is 27%. All structured outputs parse successfully. These values are recomputed from `runs/pilot_qwen25vl3b/predictions.jsonl`; the paper stores only the small aggregate evidence file.

## Verified 300-sample expansion

The non-overlapping COCO validation expansion contains 2,400 prediction rows. Strict ASR is 65.33% for causal typography, 19.67% for naive typography, and 25.33% for the plaque baseline. The paired CTA-minus-naive gap is 45.67 percentage points with a percentile 95% bootstrap interval of [39.33, 52.00]. The consistency wrapper leaves causal ASR at 65.33% while reducing naive ASR to 0%; renderer-bbox masking reduces both to 0%. Clean object accuracy is 33.67%. Manifest auditing confirms 300 unique IDs/hashes and zero image-hash overlap with the pilot. The source of record is `runs/main_qwen25vl3b_n300/predictions.jsonl` with `summary.json` and `provenance.json` in the same directory.

## Verified scale replay and practical OCR defense

The Qwen2.5-VL-7B inference-only replay contains 1,200 rows over the exact same raw rendered inputs. Clean object accuracy is 36.33%; strict ASR is 11.00% for naive typography, 20.33% for the scene-coherent plaque, and 0.00% for CTA. CTA grounded transcription is 100.00%, so the larger checkpoint reads and rejects every impossible claim in this benchmark. This is evidence of scale sensitivity, not cross-family transfer.

RapidOCR 3.9.2 detects at least half of the overlay content tokens on all 600 naive/CTA images. Mean token recall is 100.00% for naive and 99.43% for CTA; mean rectangular-mask area upper bounds are 4.53% and 14.84%. Re-evaluating those masks on the 3B checkpoint produces 0.00% strict ASR for both attacks. CTA object accuracy is 32.33%, compared with 31.33% under the renderer-box oracle. These results apply to the current high-contrast renderer and do not imply robustness to stylized or physical text.

## Configuration

YAML files control seed, sample count, local model path, image pixel budget, attacks, defenses, and output location. Generation is greedy (`do_sample=False`). `configs/smoke.yaml` uses two samples; the pilot config uses 100.

## Known limitations

1. COCO128 is a convenience subset and the largest-area label is only a proxy for the "most prominent" object. Reported object accuracy is therefore task-specific.
2. Causal text is generated from transparent class-conditioned templates. This isolates the threat mechanism but under-represents linguistic diversity.
3. The quality judge and attacked model are the same Qwen checkpoint in the first pilot; ratings are diagnostic, not independent human judgments.
4. The consistency wrapper is a lexical proxy inspired by the scene-text consistency idea, not SAGE code or a reproduction of any anonymous manuscript.
5. RapidOCR provides one practical localization experiment, but one OCR engine and high-contrast overlays do not establish robustness to stylized or physical text. Renderer-box masking remains an oracle upper bound.
6. The scene-coherent baseline changes typography and placement but is not a public SceneTAP implementation.
7. The 300-image expansion uses the verified COCO validation mirror. Cross-architecture COCO replays and Pascal VOC experiments broaden coverage but remain 300-image diagnostic samples rather than standard benchmark evaluations. Current bootstrap intervals quantify image-sampling uncertainty only.
8. The shared server environment emits a torchvision binary-extension warning due to a version mismatch. This pipeline uses PIL rather than `torchvision.io`, and the smoke/pilot runs complete, but an isolated environment should resolve the mismatch before broader reuse.
9. The natural-render comparison uses only SceneTAP's TextDiffuser component with deterministic fixed placement. It does not reproduce SceneTAP's multimodal content/placement planner and is not a photorealistic or physical-world study.
10. The blinded human protocol is implemented, but three independent response files have not yet been collected; all current quality diagnostics remain model-generated until that collection is complete.

## Public sources only

The implementation and paper cite only public model, dataset, and typographic-attack sources. Anonymous/user-provided SAGE or UDP drafts are neither copied nor cited.
