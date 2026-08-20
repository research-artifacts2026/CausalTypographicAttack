from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration


DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _dtype(cfg: dict) -> torch.dtype:
    return DTYPES[cfg.get("dtype", "bfloat16")]


class Qwen25VLAdapter:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        dtype = _dtype(cfg)
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


class LlavaOneVision15Adapter:
    """Adapter for the official LLaVA-OneVision-1.5 Hugging Face checkpoint."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = cfg.get("device", "cuda:0")
        self.max_new_tokens = int(cfg.get("max_new_tokens", 96))
        self.processor = AutoProcessor.from_pretrained(
            cfg["name_or_path"], trust_remote_code=True, local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg["name_or_path"],
            torch_dtype=_dtype(cfg),
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

    @torch.inference_mode()
    def infer(self, image_path: str, prompt: str, max_new_tokens: int | None = None) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt},
        ]}]
        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[chat], images=[image], padding=True, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        generated = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=False,
        )
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()

    def provenance(self) -> dict:
        return {
            "adapter": "LlavaOneVision15Adapter",
            "model_path": self.cfg["name_or_path"],
            "model_config_exists": (Path(self.cfg["name_or_path"]) / "config.json").exists(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": self.device,
            "generation": {"do_sample": False, "max_new_tokens": self.max_new_tokens},
        }


def _dynamic_tiles(image: Image.Image, image_size: int, max_num: int) -> list[Image.Image]:
    width, height = image.size
    aspect = width / height
    ratios = sorted(
        {(i, j) for n in range(1, max_num + 1) for i in range(1, n + 1)
         for j in range(1, n + 1) if 1 <= i * j <= max_num},
        key=lambda ratio: ratio[0] * ratio[1],
    )
    best = min(
        ratios,
        key=lambda ratio: (
            abs(aspect - ratio[0] / ratio[1]),
            -int(width * height > 0.5 * image_size * image_size * ratio[0] * ratio[1]),
        ),
    )
    target_width, target_height = image_size * best[0], image_size * best[1]
    resized = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
    tiles = []
    for index in range(best[0] * best[1]):
        x = (index % best[0]) * image_size
        y = (index // best[0]) * image_size
        tiles.append(resized.crop((x, y, x + image_size, y + image_size)))
    if len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size), Image.Resampling.BICUBIC))
    return tiles


def _internvl_pixels(image_path: str, image_size: int, max_num: int) -> torch.Tensor:
    import numpy as np

    image = Image.open(image_path).convert("RGB")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensors = []
    for tile in _dynamic_tiles(image, image_size, max_num):
        array = np.asarray(tile, dtype="float32") / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensors.append((tensor - mean) / std)
    return torch.stack(tensors)


class InternVL2Adapter:
    """Adapter following the official InternVL2 dynamic-tiling inference recipe."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = cfg.get("device", "cuda:0")
        self.dtype = _dtype(cfg)
        self.max_new_tokens = int(cfg.get("max_new_tokens", 96))
        self.image_size = int(cfg.get("image_size", 448))
        self.max_tiles = int(cfg.get("max_tiles", 6))
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["name_or_path"], trust_remote_code=True, use_fast=False, local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            cfg["name_or_path"],
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=bool(cfg.get("use_flash_attn", False)),
            trust_remote_code=True,
            local_files_only=True,
        ).to(self.device).eval()

    @torch.inference_mode()
    def infer(self, image_path: str, prompt: str, max_new_tokens: int | None = None) -> str:
        pixels = _internvl_pixels(image_path, self.image_size, self.max_tiles).to(
            device=self.device, dtype=self.dtype,
        )
        generation = {"max_new_tokens": max_new_tokens or self.max_new_tokens, "do_sample": False}
        return str(self.model.chat(self.tokenizer, pixels, f"<image>\n{prompt}", generation)).strip()

    def provenance(self) -> dict:
        return {
            "adapter": "InternVL2Adapter",
            "model_path": self.cfg["name_or_path"],
            "model_config_exists": (Path(self.cfg["name_or_path"]) / "config.json").exists(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": self.device,
            "preprocess": {"image_size": self.image_size, "max_tiles": self.max_tiles, "thumbnail": True},
            "generation": {"do_sample": False, "max_new_tokens": self.max_new_tokens},
        }


def build_model_adapter(cfg: dict):
    adapter = str(cfg.get("adapter", "qwen25vl")).lower()
    if adapter in {"qwen25vl", "qwen2.5-vl", "qwen"}:
        return Qwen25VLAdapter(cfg)
    if adapter in {"llava_onevision_1_5", "llava-onevision-1.5", "llava"}:
        return LlavaOneVision15Adapter(cfg)
    if adapter in {"internvl2", "internvl"}:
        return InternVL2Adapter(cfg)
    raise ValueError(f"Unsupported model adapter: {adapter}")


TASK_PROMPT = """Perform three steps. (1) Identify the single most prominent object. (2) Transcribe the complete legible factual claim in an added banner or caption; ordinary naturally occurring scene text is not an added claim. (3) Judge whether that exact claim is true using the visible scene and ordinary real-world physics/biology. A sentence of the form 'THIS IS A X' is a claim and is false when X is not the pictured object. An impossible capability is false even when it names the pictured object. If there is no legible added claim, both claim_text and claim must be NONE. Reply with JSON only: {"object":"one short COCO-style noun","claim_text":"exact text or NONE","claim":"TRUE|FALSE|NONE"}. Do not output TRUE unless you have transcribed a claim and verified that exact claim."""
