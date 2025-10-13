from __future__ import annotations
import os, requests
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

@dataclass
class GenConfig:
    temperature: float = 0.5
    top_p: float = 0.95
    max_new_tokens: int = 256

@dataclass
class GenOutput:
    text: str

class BaseModelClient:
    def generate(self, prompt: str, cfg: GenConfig) -> GenOutput: ...
    def nll(self, prompt: str, answer: str) -> float: ...

class HFModelClient(BaseModelClient):
    def __init__(self, model_id: str, eightbit: bool = False, flash: bool = True, cache_dir: Optional[str] = None):
        if flash:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        quant = BitsAndBytesConfig(load_in_8bit=True) if eightbit else None
        tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True) if cache_dir else AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        kwargs = {}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.bfloat16
        mdl = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, quantization_config=quant, low_cpu_mem_usage=True, **kwargs) if cache_dir else AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant, low_cpu_mem_usage=True, **kwargs)
        mdl.eval()
        self.tok, self.model, self.device = tok, mdl, mdl.device

    def generate(self, prompt: str, cfg: GenConfig) -> GenOutput:
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.pad_token_id,
        )
        text = self.tok.decode(out[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):]
        return GenOutput(text=text.strip())

    def nll(self, prompt: str, answer: str) -> float:
        text = f"{prompt}{answer}"
        batch = self.tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            loss = self.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"), labels=batch["input_ids"]).loss
        return float(-loss.detach().to(torch.float32).item())

class OllamaClient(BaseModelClient):
    def __init__(self, model_id: str, base_url: str = "http://localhost:11434"):
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, cfg: GenConfig) -> GenOutput:
        r = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model_id, "prompt": prompt, "options": {"temperature": cfg.temperature, "top_p": cfg.top_p, "num_predict": cfg.max_new_tokens}, "stream": False},
            timeout=600,
        )
        r.raise_for_status()
        j = r.json()
        return GenOutput(text=(j.get("response") or "").strip())

    def nll(self, prompt: str, answer: str) -> float:
        return float("nan")

def get_model(model_id: str, backend: str = "hf", **kwargs) -> BaseModelClient:
    if backend == "hf":
        return HFModelClient(model_id, **kwargs)
    if backend == "ollama":
        return OllamaClient(model_id, **kwargs)
    raise ValueError(f"unknown backend: {backend}")

def load_llm(model_name: str, eightbit: bool = True, flash: bool = True):
    cli = HFModelClient(model_name, eightbit=eightbit, flash=flash)
    return cli.tok, cli.model
