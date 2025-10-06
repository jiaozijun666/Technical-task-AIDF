from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Literal, Any, Union
import os

# --- optional HF login (for gated models like Llama 3.1) ---------------------
# Reads token from either secrets_local.py (if you keep it out of git)
# or from environment variables HF_TOKEN / HUGGINGFACE_HUB_TOKEN.
try:
    from huggingface_hub import login as hf_login  # type: ignore
except Exception:  # huggingface-hub not installed yet
    hf_login = None  # type: ignore

def _maybe_hf_login() -> None:
    token: Optional[str] = None
    # 1) local untracked file
    try:
        from src.secrets_local import HF_TOKEN as _TOK  # type: ignore
        token = _TOK or None
    except Exception:
        pass
    # 2) env vars
    token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    if token and hf_login:
        try:
            hf_login(token=token, add_to_git_credential=False)
        except Exception:
            # best-effort login; ignore if already logged in or offline
            pass

_maybe_hf_login()

# --- torch / transformers / requests (optional for Ollama) -------------------
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    torch = None  # type: ignore
    AutoTokenizer = AutoModelForCausalLM = None  # type: ignore

try:
    import requests
except Exception:
    requests = None  # type: ignore


# ============================ Public data classes ============================

@dataclass
class GenConfig:
    temperature: float = 0.5
    top_p: float = 1.0
    top_k: int = 50
    max_new_tokens: int = 64
    do_sample: bool = True
    stop: Optional[List[str]] = None  # hard stop strings (truncation)


@dataclass
class GenOutput:
    text: str
    raw: Any = None  # raw backend return (tensor ids / HTTP payload)


# ============================== Base interface ===============================

class BaseModelClient:
    backend: str

    def generate(self, prompt: str, cfg: GenConfig) -> GenOutput:
        raise NotImplementedError

    def nll(self, prompt: str, target: str) -> float:
        """
        Mean token-level negative log-likelihood of `target` conditioned on `prompt`.
        Implemented by HF backend; Ollama backend raises NotImplementedError.
        """
        raise NotImplementedError


# ================================ HF backend =================================

class HFModelClient(BaseModelClient):
    """
    Minimal HuggingFace transformers client:
      - auto device (cuda -> mps -> cpu) unless provided
      - reads HF token from env or secrets_local (best-effort login)
      - exposes `.tokenizer` and `.model` for baseline compatibility
      - implements `nll()` for perplexity baseline
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        dtype: Optional[Union[str, "torch.dtype"]] = None,
    ):
        if torch is None or AutoTokenizer is None:
            raise RuntimeError("HF backend requires torch+transformers installed.")

        self.backend = "hf"
        self.model_id = model_id
        self.device = self._resolve_device(device)

        # dtype can be a torch dtype or a string like "float16"
        if dtype is None:
            self.dtype = torch.float16 if self.device.type in ("cuda", "mps") else torch.float32
        elif isinstance(dtype, str):
            self.dtype = getattr(torch, dtype)
        else:
            self.dtype = dtype

        # optional auth kwargs (for gated models)
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        auth_kwargs = {"token": token} if token else {}

        # tokenizer
        self.tok = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=True,
            **auth_kwargs,
        )
        # ensure pad token exists
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token = self.tok.eos_token

        # model
        self.mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **auth_kwargs,
        ).to(self.device)
        self.mdl.eval()

    # ---- public properties (baseline compatibility) ----
    @property
    def tokenizer(self):
        return self.tok

    @property
    def model(self):
        return self.mdl

    # ---- API ----
    def generate(self, prompt: str, cfg: GenConfig) -> GenOutput:
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.mdl.generate(
                **enc,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                do_sample=cfg.do_sample,
                pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.tok.eos_token_id,
            )
        gen_ids = out_ids[0][enc.input_ids.shape[1]:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True).strip()

        if cfg.stop:
            for s in cfg.stop:
                i = text.find(s)
                if i >= 0:
                    text = text[:i]
                    break

        return GenOutput(text=text, raw=out_ids)

    def nll(self, prompt: str, target: str) -> float:
        """
        Compute mean token-level NLL (cross-entropy) of `target` given `prompt`.
        Returns a Python float (lower is better).
        """
        with torch.no_grad():
            enc_prompt = self.tok(
                prompt, return_tensors="pt", add_special_tokens=False
            )
            enc_target = self.tok(
                target, return_tensors="pt", add_special_tokens=False
            )

            input_ids = torch.cat(
                [enc_prompt.input_ids, enc_target.input_ids], dim=1
            ).to(self.device)

            labels = input_ids.clone()
            # ignore prompt positions in loss
            labels[:, : enc_prompt.input_ids.shape[1]] = -100

            out = self.mdl(input_ids=input_ids, labels=labels)
            loss = out.loss  # mean CE over non-ignored labels

        return float(loss.item())

    # ---- helpers ----
    def _resolve_device(self, device: Optional[str]):
        if device and device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


# =============================== Ollama backend ===============================

class OllamaClient(BaseModelClient):
    """
    Simple Ollama client (no NLL support, generation only).
    Requires `requests`.
    """

    def __init__(self, model_id: str, api_base: Optional[str] = None):
        if requests is None:
            raise RuntimeError("Ollama backend requires the `requests` package.")
        self.backend = "ollama"
        self.model_id = model_id
        self.api = (api_base or "http://127.0.0.1:11434").rstrip("/")

    def generate(self, prompt: str, cfg: GenConfig) -> GenOutput:
        r = requests.post(
            f"{self.api}/api/generate",
            json={
                "model": self.model_id,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": cfg.temperature,
                    "top_p": cfg.top_p,
                    "top_k": cfg.top_k,
                    "num_predict": cfg.max_new_tokens,
                },
            },
            timeout=600,
        )
        r.raise_for_status()
        text = (r.json().get("response") or "").strip()

        if cfg.stop:
            for s in cfg.stop:
                i = text.find(s)
                if i >= 0:
                    text = text[:i]
                    break

        return GenOutput(text=text)

    def nll(self, prompt: str, target: str) -> float:
        raise NotImplementedError("Ollama backend does not support NLL/perplexity.")


# =============================== public factory ===============================

def get_model(
    model_id: str,
    backend: Literal["hf", "ollama"] = "hf",
    *,
    device: Optional[str] = None,
    dtype: Optional[Union[str, "torch.dtype"]] = None,
    api_base: Optional[str] = None,
) -> BaseModelClient:
    if backend == "hf":
        return HFModelClient(model_id=model_id, device=device, dtype=dtype)
    if backend == "ollama":
        return OllamaClient(model_id=model_id, api_base=api_base)
    raise ValueError(f"Unknown backend: {backend}")


__all__ = [
    "GenConfig",
    "GenOutput",
    "BaseModelClient",
    "HFModelClient",
    "OllamaClient",
    "get_model",
]
