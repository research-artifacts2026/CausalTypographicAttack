# SCEI-Images-300 four-model victim evaluation

This directory contains the complete path-free aggregate release for the frozen
SCEI-Images-300 victim evaluation. The manifest was fixed before victim
inference. Each model has 300 items and five conditions (`clean_false`,
`clean_true`, `flat_false`, `scene_false`, and `scene_true`), for exactly 1,500
rows per model.

Strict grounded success requires all three gates: the same model answers the
clean false-record question correctly, changes the attacked answer to `YES`,
and exactly transcribes the complete registered measurement and uncertainty
under a separate read prompt. Rates below use each model's own clean-correct
population.

| Model | Clean-correct n | Flat strict | Scene strict | Scene-flat | Scene target | Scene exact read | True control |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B | 187 | 11.8% | 11.8% | 0.0 pp | 75.4% | 16.6% | 88.8% |
| Qwen2.5-VL-7B | 253 | 17.8% | 15.4% | -2.4 pp | 51.0% | 19.8% | 100.0% |
| LLaVA-OneVision-1.5-8B | 237 | 4.2% | 2.1% | -2.1 pp | 12.2% | 28.3% | 100.0% |
| InternVL2-8B | 184 | 7.1% | 10.9% | +3.8 pp | 55.4% | 22.3% | 97.3% |

The attack induces nonzero strict grounded failures on every checkpoint, but
scene integration does not improve aggregate success over matched flat text.
Exact two-sided McNemar p-values are 1.000, 0.377, 0.227, and 0.210 in table
order. The exact-read gate, not only target influence, is the main bottleneck.

## Files

- `public_analysis.json`: all model, family, answer-cell, Wilson-interval, and
  exact paired results, with raw-log/provenance hashes and no private paths;
- `model_summary.csv`, `family_summary.csv`, and `table.tex`: generated views
  of the same complete aggregate.

The 6,000 sample-level answer/transcription rows remain in the audited server
and local evidence archive; they are not included in this public commit. The
analysis code also provides `validate_scei_image_eval_release.py` for validating
a locally generated path-free raw release against this aggregate.

This is a fixed digital compositing experiment. It is not physical-world,
human-deceptiveness, official SceneTAP-planner, or public-SOTA evidence. The
scene renderer is deterministic perspective/tone/texture/shadow adaptation,
not diffusion inpainting or generative-AI background editing.
