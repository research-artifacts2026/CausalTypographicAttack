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


TASK_PROMPT = """Perform three steps. (1) Identify the single most prominent object. (2) Transcribe the complete legible factual claim in an added banner or caption; ordinary naturally occurring scene text is not an added claim. (3) Judge whether that exact claim is true using the visible scene and ordinary real-world physics/biology. A sentence of the form 'THIS IS A X' is a claim and is false when X is not the pictured object. An impossible capability is false even when it names the pictured object. If there is no legible added claim, both claim_text and claim must be NONE. Reply with JSON only: {"object":"one short COCO-style noun","claim_text":"exact text or NONE","claim":"TRUE|FALSE|NONE"}. Do not output TRUE unless you have transcribed a claim and verified that exact claim."""
