"""Hugging Face Spaces / local entry point for the SCEI-Search Gradio UI.

The UI itself is model-agnostic.  Model checkpoints and devices are selected
through ``SCEI_DEMO_CONFIG`` so credentials and machine-specific paths never
need to be embedded in this file.
"""

from __future__ import annotations

import os
from pathlib import Path

from scripts.launch_scei_gradio import build_demo


ROOT = Path(__file__).resolve().parent
CONFIG = Path(os.environ.get("SCEI_DEMO_CONFIG", ROOT / "configs" / "scei_gradio_local_v1.yaml"))
OUTPUT = Path(os.environ.get("SCEI_DEMO_OUTPUT", ROOT / "runs" / "scei_gradio_sessions"))

demo = build_demo(CONFIG.resolve(), OUTPUT.resolve())


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=os.environ.get("GRADIO_SHARE", "0") == "1",
        show_error=True,
    )
