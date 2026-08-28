# Balanced CTA experiment extension (2026-08-28)

This extension adds two newer open LVLM adapters, an attack-oblivious semantics-first defense diagnostic, and a text/geometry-matched delivery-layer comparison. All reported values are generated from complete logs; the table generator stops if the required provenance is incomplete.

## Frozen protocol

- Source: the existing 300-item balanced COCO manifest (`runs/rvtaqa_balanced_coco_n300/render_manifest.jsonl`).
- Attack conditions: no attack, benign control, direct answer, Plain Claim, Evidence CTA, and Causal-Bridge.
- Counterbalancing: truth direction, A/B order, and semantic YES/NO format are assigned before inference.
- Grounded success: the model must be clean-correct, reach the registered attacked target, and exactly transcribe the registered claim in a separate query.
- Decoding: greedy, one answer query plus one independent transcription query, without response-conditioned regeneration.

## New checkpoint replay

| Model | Items | Clean | Plain macro | Evidence macro | Bridge macro | Paired Bridge−Plain | Exact McNemar |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-8B | 300 | 82.7% | 17.4% | 14.0% | 26.7% | +8.9 pp | 5.95e-5 |
| InternVL3-8B | 300 | 80.0% | 37.2% | 42.7% | 48.1% | +12.5 pp | 9.25e-6 |

The macro columns are unweighted over the six counterbalance cells. The paired differences use the pooled clean-correct population. These are COCO-only checkpoint updates, not second-dataset or public-leaderboard results.

## Semantic prompt defense

The wrapper sees the same attacked pixels and question as the base model. It receives no OCR boxes or attack label; it is told to treat in-image text as an untrusted claim and verify it against the typed ordinary-world assumptions.

| Model | Base clean | Defended clean | Base Bridge | Defended Bridge | Exact McNemar |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 80.3% | 69.0% | 69.9% | 6.8% | 1.47e-39 |
| LLaVA-OneVision-1.5-8B | 82.3% | 80.0% | 47.9% | 13.3% | 3.80e-22 |

Bridge rates use the shared-clean population. The wrapper is a prompt diagnostic, not a trained verifier or a reproduction of SAGE, and its clean-accuracy cost must be reported.

## Matched delivery-layer replay

For every item, flat and scene-integrated branches have identical title, false proposition, direction-conditioned conclusion, status, bounding box, font geometry, question, and transcription target. Only deterministic local tone, perspective, and shadow differ.

| Model | Clean-correct | Flat Bridge | Scene-integrated Bridge | Difference | Exact McNemar |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 241 | 59.3% | 64.3% | +5.0 pp | .073 |
| LLaVA-OneVision-1.5-8B | 247 | 39.3% | 39.3% | 0.0 pp | 1.000 |

The renderer preserves substantial attack success but does not demonstrate a statistically significant general improvement. It is deterministic synthetic compositing, not camera capture, AI inpainting, physical transfer, or human-rated naturalness.

## Reproduction

Run from the repository root after editing the model snapshot paths in the selected YAML file:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_rvta_qa_balanced.py \
  --config configs/rvtaqa_balanced_coco_qwen3vl8_n300.yaml

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/compatible/internvl/site-packages \
  python scripts/run_rvta_qa_balanced.py \
  --config configs/rvtaqa_balanced_coco_internvl3_8b_n300.yaml

CUDA_VISIBLE_DEVICES=0 python scripts/run_semantic_prompt_defense.py \
  --config configs/semantic_defense_qwen7_coco_n300.yaml

python scripts/build_matched_bridge_renderer.py \
  --source-manifest runs/rvtaqa_balanced_coco_n300/render_manifest.jsonl \
  --output-root runs/matched_bridge_renderer_coco_n300 --limit 300

CUDA_VISIBLE_DEVICES=0 python scripts/run_matched_bridge_renderer.py \
  --config configs/matched_bridge_renderer_qwen7_coco_n300.yaml
```

For a safely interrupted long run, `shard_remaining_balanced.py` freezes the completed prefix and partitions only unfinished item keys. After all disjoint shards finish, `merge_balanced_shards.py` performs an exact union and rejects duplicates, missing or extra keys, and image-hash mismatches.

Generate release tables only after placing complete summary/provenance pairs under the result root:

```bash
python scripts/make_extension_tables.py \
  --result-root results --output-root generated
```

The generated evidence ledger records SHA-256 hashes for every input and output. No credentials are stored by these scripts.
