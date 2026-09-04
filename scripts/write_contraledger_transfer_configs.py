#!/usr/bin/env python3
"""Write the fixed dataset/method/model run matrix for the transfer study."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


MODELS = {
    "qwen3": {
        "adapter": "qwen25vl",
        "name_or_path": "/home/fangxinyue/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3",
        "device": "cuda:0", "dtype": "bfloat16", "min_pixels": 200704,
        "max_pixels": 602112, "max_new_tokens": 96, "do_sample": False,
    },
    "qwen7": {
        "adapter": "qwen25vl",
        "name_or_path": "/disk2/fangxinyue/SpaceDrive/ckpts/Qwen2.5-VL-7B-Instruct",
        "device": "cuda:0", "dtype": "bfloat16", "min_pixels": 200704,
        "max_pixels": 602112, "max_new_tokens": 96, "do_sample": False,
    },
    "llava": {
        "adapter": "llava_onevision_1_5",
        "name_or_path": "/home/fangxinyue/.cache/huggingface/hub/models--lmms-lab--LLaVA-OneVision-1.5-8B-Instruct/snapshots/e137910552d616eb0f7305147ca054311a90c542",
        "device": "cuda:0", "dtype": "bfloat16", "max_new_tokens": 96,
        "do_sample": False,
    },
    "internvl": {
        "adapter": "internvl2",
        "name_or_path": "/home/fangxinyue/.cache/huggingface/hub/models--OpenGVLab--InternVL2-8B/snapshots/6fb9ad6924f69424e57fab2ab061d707688f0296",
        "device": "cuda:0", "dtype": "bfloat16", "max_new_tokens": 96,
        "do_sample": False, "image_size": 448, "max_tiles": 6,
        "use_flash_attn": False,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--expected-items", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    matrices = {
        "coco_native": "runs/contraledger_threeway_n200_v1frozen/manifest.jsonl",
        "coco_flat": "runs/contraledger_threeway_coco_n200_flat_v1/manifest.jsonl",
        "coco_scenetap": "runs/contraledger_threeway_coco_n200_scenetap_v1/manifest.jsonl",
        "voc_native": "runs/contraledger_threeway_voc2012_n200_v1frozen/manifest.jsonl",
        "voc_flat": "runs/contraledger_threeway_voc2012_n200_flat_v1/manifest.jsonl",
        "voc_scenetap": "runs/contraledger_threeway_voc2012_n200_scenetap_v1/manifest.jsonl",
    }
    args.config_dir.mkdir(parents=True, exist_ok=True)
    for cell, manifest in matrices.items():
        dataset, method = cell.split("_", 1)
        for model_name, model in MODELS.items():
            payload = {
                "source_manifest": manifest,
                "output_root": f"runs/contraledger_delivery_{dataset}_{method}_{model_name}_n200_v1",
                "expected_items": args.expected_items,
                "seed": args.seed,
                "model": model,
            }
            path = args.config_dir / f"contraledger_delivery_{dataset}_{method}_{model_name}_n200.yaml"
            path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"wrote {len(matrices) * len(MODELS)} configs to {args.config_dir}")


if __name__ == "__main__":
    main()
