from __future__ import annotations
import os
import torch
from dataclasses import dataclass
from typing import Optional, Union, Literal, Any
try:
    import requests
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    raise RuntimeError("Please install transformers and requests before running.")

@dataclass
class GenConfig:
    temperature: float = 0.5
    top_p: float = 1.0
    top_k: int = 50
    max_new_tokens: int = 64
    do_sample: bool = True

@dataclass
class GenOutput:
    text: str
    raw: Any = None

class BaseModelClient:
    backend: str
    def generate(self, prompt: str, cfg: GenConfig) -> GenOutput:
        raise NotImplementedError

class HFModelClient(BaseModelClient):
    def __init__(self, model_id: str, device: Optional[str] = None):
        self.backend = "hf"
        self.model_id = model_id
        self.device = self._resolve_device(device)
        print(f"[INFO] Loading {model_id} on {self.device}...")

        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.mdl = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)
        self.mdl.eval()

    def generate(self, prompt: str, cfg: GenConfig) -> GenOutput:
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.mdl.generate(**enc, **cfg.__dict__, pad_token_id=self.tok.pad_token_id)
        gen_text = self.tok.decode(out[0], skip_special_tokens=True)
        return GenOutput(text=gen_text)

    def _resolve_device(self, device: Optional[str]):
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

class OllamaClient(BaseModelClient):
    def __init__(self, model_id: str, api_base: str = "http://127.0.0.1:11434"):
        self.backend = "ollama"
        self.model_id = model_id
        self.api = api_base

    def generate(self, prompt: str, cfg: GenConfig) -> GenOutput:
        res = requests.post(
            f"{self.api}/api/generate",
            json={"model": self.model_id, "prompt": prompt, "stream": False},
            timeout=600
        )
        res.raise_for_status()
        text = res.json().get("response", "").strip()
        return GenOutput(text=text)

def get_model(model_id: str, backend: Literal["hf", "ollama"] = "hf") -> BaseModelClient:
    if backend == "hf":
        return HFModelClient(model_id)
    elif backend == "ollama":
        return OllamaClient(model_id)
    else:
        raise ValueError(f"Unknown backend: {backend}")
