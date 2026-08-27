# SCEI-Search Adaptive Attack Lab

This Gradio interface visualizes a bounded black-box version of
**Scene-Conditioned Counterfactual Evidence Implantation (SCEI)**.

The interface shows:

- the clean gate and clean victim answer;
- every generated text/carrier candidate;
- the attacked image, victim answer, and independent transcription result;
- the registered false record that remains fixed across rounds;
- a four-state failure diagnosis and the next permitted design change;
- the round at which strict success first occurs;
- the exact planner/victim query counts and a downloadable audit bundle.

Run on the GPU server from the repository root:

```bash
/disk2/fangxinyue/.venv/bin/pip install -r requirements-gradio.txt
/disk2/fangxinyue/.venv/bin/python app.py
```

The default configuration loads a Qwen2.5-VL-3B planner on GPU 2 and a
Qwen2.5-VL-7B victim on GPU 3. Use an SSH tunnel or an approved reverse proxy
instead of exposing the research server directly.

The same loop can be audited without the web UI:

```bash
/disk2/fangxinyue/.venv/bin/python scripts/run_scei_adaptive_demo.py \
  --config configs/scei_gradio_local_v1.yaml \
  --image /absolute/path/to/image.jpg --target-label airplane \
  --output-root runs/scei_adaptive_demo_airplane_v1 --max-rounds 6
```

## Scientific boundary

Later rounds observe earlier victim answers, so this UI demonstrates an
**adaptive query attack**. It is separate from the paper's frozen transfer
tables. A successful UI run is not an aggregate result, and the deterministic
scene carrier is not diffusion inpainting or physical capture.

## Hugging Face Spaces entry point

`app.py` is a standard Gradio entry point. A Space or another machine must
provide locally accessible victim/planner checkpoints through a YAML file and
set `SCEI_DEMO_CONFIG` to that file. The default YAML contains server-local
checkpoint paths and is therefore intended for the 212 GPU server, not a
public CPU Space. Never bake an API key or model-hub token into the Space.
