---
title: SCEI-Search Adaptive Attack Lab
emoji: 🔁
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
suggested_hardware: a10g-large
models:
  - Qwen/Qwen2.5-VL-3B-Instruct
  - Qwen/Qwen2.5-VL-7B-Instruct
---

# SCEI-Search Adaptive Attack Lab

This Space visualizes **Scene-Conditioned Counterfactual Evidence
Implantation (SCEI)** as a bounded black-box search. For an uploaded image, it
keeps one false numerical record and one verification question fixed while it
may change the record's title, scene anchor, carrier style, and placement. The
trace shows the clean answer, immutable false record, every attempted
rendering, victim answer, independent transcription check, four-state failure
diagnosis, next permitted intervention, and query count.

## Deployment

The wrapper intentionally imports `build_demo` from
`scripts/launch_scei_gradio.py`; it does not maintain a second copy of the
attack or UI logic. Publish the complete `CausalTypographicAttack` checkout as
the Space source and place this folder's `app.py`, `README.md`, and
`requirements.txt` at the Space root. Direct execution from this demo folder
inside the checkout also works. If the checkout is elsewhere, set the Space
variable `SCEI_REPO_ROOT` to its absolute path.

Live inference requires a **GPU Space** and locally accessible planner and
victim checkpoints. Provide a YAML model configuration and set:

```text
SCEI_DEMO_CONFIG=/absolute/path/to/space-models.yaml
SCEI_DEMO_OUTPUT=/data/scei_gradio_sessions
```

The repository's default `configs/scei_gradio_local_v1.yaml` contains
machine-specific multi-GPU checkpoint paths and is only an example. A hosted
Space must replace those paths and device assignments with values valid for
its hardware. Do not place model-hub tokens or other secrets in the YAML or
repository; use the Space's secret settings when a gated checkpoint requires
authentication.

For local verification from the repository root:

```bash
pip install -r demo/scei_gradio_space/requirements.txt
python demo/scei_gradio_space/app.py
```

## Evidence boundary

This UI is an **adaptive demonstration**: after the first attempt, later
designs may use earlier victim answers and transcription outcomes. Its retries
therefore are not evidence for a frozen, zero-feedback transfer evaluation.
Do not merge a successful interactive trace into fixed-evaluation ASR. Report
adaptive runs separately with Success@K, queries to success, all exhausted
budgets, and the downloadable audit trace produced by the interface.

A visually integrated carrier also is not, by itself, evidence of physical
realism or human-perceived naturalness.
