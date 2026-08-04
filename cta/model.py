from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class Qwen25VLAdapter:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.get("dtype", "bfloat16")]
        self.processor = AutoProcessor.from_pretrained(
            cfg["name_or_path"], local_files_only=True,
            min_pixels=cfg.get("min_pixels"), max_pixels=cfg.get("max_pixels"),
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg["name_or_path"], torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True,
        ).to(cfg.get("device", "cuda:0")).eval()
        self.device = cfg.get("device", "cuda:0")
        self.max_new_tokens = int(cfg.get("max_new_tokens", 96))

    @torch.inference_mode()
    def infer(self, image_path: str, prompt: str, max_new_tokens: int | None = None) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=False)
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def provenance(self) -> dict:
        cfg_path = Path(self.cfg["name_or_path"]) / "config.json"
        return {
            "adapter": "Qwen25VLAdapter",
            "model_path": self.cfg["name_or_path"],
            "model_config_exists": cfg_path.exists(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": self.device,
            "generation": {"do_sample": False, "max_new_tokens": self.max_new_tokens},
        }


TASK_PROMPT = """Identify the single most prominent object in the image. Also inspect any printed factual claim added as a caption or banner. Reply exactly as JSON: {"object": "one short COCO-style noun", "claim": "TRUE|FALSE|NONE"}. Use NONE only when no added factual claim is present. Judge claim truth using both the visible scene and ordinary real-world physics/biology, not merely image-text consistency."""

