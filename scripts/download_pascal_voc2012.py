#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    root = Path("/disk2/fangxinyue/datasets/pascal_voc_2012_hf")
    root.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id="nateraw/pascal-voc-2012", repo_type="dataset",
        allow_patterns=["data/*.parquet"], local_dir=str(root),
    )
    print(path)


if __name__ == "__main__":
    main()
