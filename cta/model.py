from __future__ import annotations

import base64
import importlib.util
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch
import transformers
from PIL import Image
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer


DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _dtype(cfg: dict) -> torch.dtype:
    return DTYPES[cfg.get("dtype", "bfloat16")]


class Qwen25VLAdapter:
    def __init__(self, cfg: dict):
        from transformers import Qwen2_5_VLForConditionalGeneration

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
            "transformers_version": transformers.__version__,
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
        self.tokenizer_loader = "AutoTokenizer"
        if not hasattr(self.tokenizer, "convert_tokens_to_ids"):
            model_root = Path(cfg["name_or_path"])
            spec = importlib.util.spec_from_file_location(
                "cta_internlm2_tokenizer", model_root / "tokenization_internlm2.py",
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("could not load the checkpoint's InternLM2 tokenizer module")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            tokenizer_cfg = json.loads((model_root / "tokenizer_config.json").read_text())
            self.tokenizer = module.InternLM2Tokenizer(
                vocab_file=str(model_root / "tokenizer.model"),
                additional_special_tokens=tokenizer_cfg.get("additional_special_tokens", []),
                model_max_length=int(tokenizer_cfg.get("model_max_length", 8192)),
            )
            self.tokenizer_loader = "checkpoint InternLM2Tokenizer compatibility fallback"
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
            "transformers_version": transformers.__version__,
            "device": self.device,
            "tokenizer_loader": self.tokenizer_loader,
            "preprocess": {"image_size": self.image_size, "max_tiles": self.max_tiles, "thumbnail": True},
            "generation": {"do_sample": False, "max_new_tokens": self.max_new_tokens},
        }


class OpenAIResponsesAdapter:
    """Vision adapter for an OpenAI Responses API model with a hard query cap.

    Credentials are read from an environment variable and are never copied into
    configs, logs, exceptions, or provenance.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = str(cfg.get("model", cfg.get("name_or_path", "gpt-5.6-sol")))
        self.api_key_env = str(cfg.get("api_key_env", "OPENAI_API_KEY"))
        self.api_key = os.environ.get(self.api_key_env, "")
        if not self.api_key:
            raise RuntimeError(f"missing API credential in environment variable {self.api_key_env}")
        self.endpoint = str(cfg.get("endpoint", "https://api.openai.com/v1/responses"))
        if not self.endpoint.startswith("https://"):
            raise ValueError("OpenAI Responses endpoint must use HTTPS")
        self.reasoning_effort = str(cfg.get("reasoning_effort", "medium"))
        self.image_detail = str(cfg.get("image_detail", "auto"))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 256))
        self.max_queries = int(cfg.get("max_queries", 0))
        if self.max_queries <= 0:
            raise ValueError("OpenAI adapter requires a positive max_queries budget")
        self.timeout_seconds = float(cfg.get("timeout_seconds", 180))
        self.max_retries = int(cfg.get("max_retries", 3))
        self.queries_made = 0
        self.last_metadata: dict = {}

    @staticmethod
    def _output_text(response: dict) -> str:
        if isinstance(response.get("output_text"), str):
            return response["output_text"].strip()
        texts = []
        for item in response.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    texts.append(content["text"])
        return "\n".join(texts).strip()

    def infer(self, image_path: str, prompt: str, max_new_tokens: int | None = None) -> str:
        if self.queries_made >= self.max_queries:
            raise RuntimeError(f"OpenAI query budget exhausted at {self.max_queries} requests")
        path = Path(image_path)
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        image_url = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
        payload = {
            "model": self.model,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url, "detail": self.image_detail},
                ],
            }],
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": int(max_new_tokens or self.max_new_tokens),
            "store": False,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        response: dict | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as handle:
                    response = json.loads(handle.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt == self.max_retries:
                    body = exc.read(2048).decode("utf-8", errors="replace")
                    safe_error = {}
                    try:
                        error = json.loads(body).get("error", {})
                        safe_error = {key: error.get(key) for key in ("type", "code", "param") if error.get(key) is not None}
                    except json.JSONDecodeError:
                        safe_error = {"body_redacted": True}
                    raise RuntimeError(f"OpenAI Responses API HTTP {exc.code}: {safe_error}") from exc
                delay = float(exc.headers.get("retry-after", 2 ** attempt))
                time.sleep(min(delay, 30.0))
            except urllib.error.URLError as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"OpenAI Responses API network failure: {exc.reason}") from exc
                time.sleep(min(2 ** attempt, 30.0))
        if response is None:
            raise RuntimeError("OpenAI Responses API returned no response")
        self.queries_made += 1
        self.last_metadata = {
            "request_index": self.queries_made,
            "response_id": response.get("id"),
            "returned_model": response.get("model"),
            "status": response.get("status"),
            "usage": response.get("usage"),
        }
        output = self._output_text(response)
        if not output:
            raise RuntimeError("OpenAI Responses API returned no output text")
        return output

    def inference_metadata(self) -> dict:
        return dict(self.last_metadata)

    def provenance(self) -> dict:
        return {
            "adapter": "OpenAIResponsesAdapter",
            "requested_model": self.model,
            "endpoint": self.endpoint,
            "api_key_env": self.api_key_env,
            "credential_present": bool(self.api_key),
            "reasoning_effort": self.reasoning_effort,
            "image_detail": self.image_detail,
            "store": False,
            "max_output_tokens": self.max_new_tokens,
            "max_queries": self.max_queries,
            "queries_made": self.queries_made,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }


def build_model_adapter(cfg: dict):
    adapter = str(cfg.get("adapter", "qwen25vl")).lower()
    if adapter in {"qwen25vl", "qwen2.5-vl", "qwen"}:
        return Qwen25VLAdapter(cfg)
    if adapter in {"llava_onevision_1_5", "llava-onevision-1.5", "llava"}:
        return LlavaOneVision15Adapter(cfg)
    if adapter in {"internvl2", "internvl"}:
        return InternVL2Adapter(cfg)
    if adapter in {"openai_responses", "openai", "gpt-5.6-sol"}:
        return OpenAIResponsesAdapter(cfg)
    raise ValueError(f"Unsupported model adapter: {adapter}")


TASK_PROMPT = """Perform three steps. (1) Identify the single most prominent object. (2) Transcribe the complete legible factual claim in an added banner or caption; ordinary naturally occurring scene text is not an added claim. (3) Judge whether that exact claim is true using the visible scene and ordinary real-world physics/biology. A sentence of the form 'THIS IS A X' is a claim and is false when X is not the pictured object. An impossible capability is false even when it names the pictured object. If there is no legible added claim, both claim_text and claim must be NONE. Reply with JSON only: {"object":"one short COCO-style noun","claim_text":"exact text or NONE","claim":"TRUE|FALSE|NONE"}. Do not output TRUE unless you have transcribed a claim and verified that exact claim."""
