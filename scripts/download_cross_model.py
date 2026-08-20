#!/usr/bin/env python3
"""Download a public cross-family checkpoint into the Hugging Face cache."""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download


PUBLIC_MODELS = {
    "internvl2-8b": "OpenGVLab/InternVL2-8B",
    "llava-onevision-1.5-8b": "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(PUBLIC_MODELS))
    args = parser.parse_args()
    path = snapshot_download(repo_id=PUBLIC_MODELS[args.model])
    print(path)


if __name__ == "__main__":
    main()
