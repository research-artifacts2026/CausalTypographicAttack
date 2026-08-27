# SCEI-Images-300

This image-first dataset contains 300 source scenes and
900 portable images. Every source scene has an exact
clean copy, a scene-adaptive false counterfactual carrier, and a geometry- and
layout-matched corrected carrier.

## Variants

- `clean`: an exact copy of the registered source image.
- `attack_false`: one mechanically invalid field on a scene-adaptive carrier.
- `control_true`: the corresponding corrected field on the same carrier geometry.

## Counterfactual families

- `capacity_conservation`: 38
- `causal_order`: 37
- `geometry_feasibility`: 37
- `phase_state`: 37
- `probability_ledger`: 37
- `range_threshold`: 38
- `temporal_ledger`: 38
- `unit_conversion`: 38

## Important boundary

The carrier uses deterministic perspective, local color, texture, placement,
and shadow adaptation. It is synthetic compositing, not diffusion inpainting,
physical capture, or proof of attack success. Attack success must be measured
separately against a frozen victim protocol.

See `manifest.jsonl`, `selection.jsonl`, and `provenance.json` for registered
claims, hashes, masks, source paths, and generation metadata.
