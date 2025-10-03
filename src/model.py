# coding: utf-8
"""
Unified model client for all baselines.

- HF backend: transformers + torch (supports generate + nll/perplexity)
- Ollama backend: HTTP API to local server (supports generate only)

Usage:
    from src.model import get_model, GenConfig

    mdl = get_model("Qwen/Qwen2.5-1.5B-Instruct", backend="hf")
    out = mdl.generate("Say hi", GenConfig(max_new_tokens=16, temperature=0.5))
    print(out.text)

    # NLL / perplexity (HF only)
    n = mdl.nll("Question: ...\nAnswer:", "short answer")
    print(n.loss, n.perplexity)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Literal
import math

# ---- optional deps guarded to keep import cheap for Ollama-only runs
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    torch = None
    AutoTokenizer = AutoModelForCausalLM = None

# ---- small HTTP client for Ollama
try:
    import requests
except Exception:
    requests = None


# ----------------------- configs & small structs -----------------------

@dataclass
class GenConfig:
    temperature: float = 0.5
    top_p: float = 1.0
    top_k: int = 50
    max_new_tokens: int = 64
    do_sample: bool = True
    stop: Optional[List[str]] = None   # optional list of stop strings


@dataclass
class GenOutput:
    text: str
    raw: Any = None


@dataclass
class NLLResult:
    loss: float             # mean loss over target tokens
    tokens: int             # # target tokens
    nll: float              # loss * tokens
    perplexity: float


# ----------------------- base client -----------------------

class BaseModelClient:
    backend: str

    def generate(self, prompt: str, cfg: GenConfig) -> GenOutput:
        raise NotImplementedError

    # prompt = context, target = answer text whose loss you want
    def nll(self, prompt: str, target: str) -> NLLResult:
        raise NotImplementedError


# ----------------------- HF client -----------------------

class HFModelClient(BaseModelClient):
    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        if torch is None or AutoTokenizer is None:
            raise RuntimeError("HF backend requires torch+transformers installed.")

        self.backend = "hf"
        self.model_id = model_id

        # device / dtype
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        if dtype is None:
            dtype = "float16" if device in ("cuda", "mps") else "float32"
        self.dtype = getattr(torch, dtype)

        # load
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        # some tokenizers miss pad; align with eos to avoid warnings
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token = self.tok.eos_token

        self.mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
        )
        self.mdl.to(self.device)
        self.mdl.eval()

    def generate(self, prompt: str, cfg: GenConfig) -> GenOutput:
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        gen_kwargs = dict(
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=cfg.do_sample,
            top_k=cfg.top_k,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        with torch.no_grad():
            out_ids = self.mdl.generate(**enc, **gen_kwargs)
        gen_ids = out_ids[0][enc.input_ids.shape[1]:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True)

        # stop strings handling
        if cfg.stop:
            for s in cfg.stop:
                idx = text.find(s)
                if idx >= 0:
                    text = text[:idx]
                    break
        return GenOutput(text=text.strip(), raw=out_ids)

    def nll(self, prompt: str, target: str) -> NLLResult:
        """Mean negative log-likelihood over target tokens only."""
        enc_p = self.tok(prompt, return_tensors="pt")
        enc_t = self.tok(target, add_special_tokens=False, return_tensors="pt")
        input_ids = torch.cat([enc_p["input_ids"], enc_t["input_ids"]], dim=1)
        attn = torch.ones_like(input_ids)

        labels = input_ids.clone()
        # ignore prompt tokens in the loss
        p_len = enc_p["input_ids"].shape[1]
        labels[:, :p_len] = -100

        input_ids = input_ids.to(self.device)
        attn = attn.to(self.device)
        labels = labels.to(self.device)

        with torch.no_grad():
            out = self.mdl(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = float(out.loss.item())  # mean over target tokens
        tgt_tokens = int(enc_t["input_ids"].numel())
        return NLLResult(loss=loss, tokens=tgt_tokens, nll=loss * tgt_tokens, perplexity=math.exp(loss))


# ----------------------- Ollama client -----------------------

class OllamaClient(BaseModelClient):
    def __init__(self, model_id: str, api_base: Optional[str] = None):
        if requests is None:
            raise RuntimeError("Ollama backend requires `requests` installed.")
        self.backend = "ollama"
        self.model_id = model_id
        self.api = api_base.rstrip("/") if api_base else "http://127.0.0.1:11434"

    def generate(self, prompt: str, cfg: GenConfig) -> GenOutput:
        url = f"{self.api}/api/generate"
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "num_predict": cfg.max_new_tokens,
            },
        }
        r = requests.post(url, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        text = (data.get("response") or "").strip()

        # apply stop strings client-side if needed
        if cfg.stop:
            for s in cfg.stop:
                idx = text.find(s)
                if idx >= 0:
                    text = text[:idx]
                    break
        return GenOutput(text=text, raw=data)

    def nll(self, prompt: str, target: str) -> NLLResult:
        raise NotImplementedError("Ollama backend does not expose token-level loss/probabilities.")


# ----------------------- factory -----------------------

def get_model(
    model_id: str,
    backend: Literal["hf", "ollama"] = "hf",
    *,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    api_base: Optional[str] = None,
) -> BaseModelClient:
    """
    Create a unified model client.

    backend='hf'    -> HuggingFace transformers (generate + nll)
    backend='ollama'-> local Ollama server (generate only)
    """
    if backend == "hf":
        return HFModelClient(model_id=model_id, device=device, dtype=dtype)
    elif backend == "ollama":
        return OllamaClient(model_id=model_id, api_base=api_base)
    else:
        raise ValueError(f"Unknown backend: {backend}")
