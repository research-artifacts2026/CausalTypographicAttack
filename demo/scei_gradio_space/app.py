"""Hugging Face Spaces entry point for the SCEI-Search adaptive demo.

The interface and experiment logic live in ``scripts/launch_scei_gradio.py``.
This file deliberately stays as a thin deployment wrapper so that local,
server, and Space runs execute the same implementation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _repository_root() -> Path:
    """Locate a checkout that contains the shared Gradio launcher."""

    candidates: list[Path] = []
    configured_root = os.environ.get("SCEI_REPO_ROOT")
    if configured_root:
        candidates.append(Path(configured_root).expanduser())
    # ``HERE`` covers the normal Space layout, where this wrapper is copied to
    # the repository root. Parents cover direct execution from this demo folder.
    candidates.extend([HERE, *HERE.parents])

    for candidate in candidates:
        if (candidate / "scripts" / "launch_scei_gradio.py").is_file():
            return candidate.resolve()

    raise RuntimeError(
        "Could not find scripts/launch_scei_gradio.py. Deploy this wrapper "
        "with the CausalTypographicAttack repository, or set SCEI_REPO_ROOT "
        "to that repository checkout."
    )


REPO_ROOT = _repository_root()
sys.path.insert(0, str(REPO_ROOT))

from scripts.launch_scei_gradio import build_demo  # noqa: E402


CONFIG = Path(
    os.environ.get(
        "SCEI_DEMO_CONFIG",
        REPO_ROOT / "configs" / "scei_gradio_local_v1.yaml",
    )
).expanduser()
OUTPUT_ROOT = Path(
    os.environ.get(
        "SCEI_DEMO_OUTPUT",
        REPO_ROOT / "runs" / "scei_gradio_space_sessions",
    )
).expanduser()

demo = build_demo(CONFIG.resolve(), OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )
