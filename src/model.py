from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Literal, Any, Union
import os

def _maybe_hf_login() -> None:
    try:
        from huggingface_hub import login as hf_login
    except ImportError:
        print("[WARN] huggingface_hub not installed, skipping login.")
        return

    token = None
    # Try secrets_local.py first
    try:
        from src.secrets_local import HF_TOKEN as _TOK  # type: ignore
        token = _TOK or None
    except Exception:
        pass

    # Fallback: environment variables
    token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    if token:
        try:
            hf_login(token)
            print("[INFO] Logged in to Hugging Face Hub.")
        except Exception as e:
            print(f"[WARN] HF login skipped: {e}")



try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    raise RuntimeError("You must install torch and transformers first.")


try:
    import requests
except ImportError:
    requests = None



@dataclass
class GenConfig:
    temperature: float = 0.5
    top_p: float = 1.0
    top_k: int = 50
    max_new_tokens: int = 64
    do_sample: bool = True
    stop: Optional[List[str]] = None


@dataclass
class GenOutput:
    text: str
    raw: Any = None

class BaseModelClient:
    backend: str

    def generate(self, prompt: str, cfg: GenConfig) -> GenOutput:
        raise NotImplementedError

    def nll(self, prompt: str, target: str) -> float:
        raise NotImplementedError



class HFModelClient(BaseModelClient):
    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        dtype: Optional[Union[str, "torch.dtype"]] = None,
        auto_login: bool = True,
    ):
        self.backend = "hf"
        self.model_id = model_id

        if auto_login:
            _maybe_hf_login()

        # Choose device
        self.device = self._resolve_device(device)
        if dtype is None:
            self.dtype = torch.float16 if self.device.type in ("cuda", "mps") else torch.float32
        elif isinstance(dtype, str):
            self.dtype = getattr(torch, dtype)
        else:
            self.dtype = dtype

        # Auth token
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        auth_kwargs = {"token": token} if token else {}

        # Tokenizer
        self.tok = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=True,
            **auth_kwargs,
        )
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        # Model
        try:
            self.mdl = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **auth_kwargs,
            )
        except TypeError:
            self.mdl = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                **auth_kwargs,
            )
        self.mdl.to(self.device)
        self.mdl.eval()

    # --- API ---
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
        enc_prompt = self.tok(prompt, return_tensors="pt", add_special_tokens=False)
        enc_target = self.tok(target, return_tensors="pt", add_special_tokens=False)
        input_ids = torch.cat([enc_prompt.input_ids, enc_target.input_ids], dim=1).to(self.device)

        labels = input_ids.clone()
        labels[:, :enc_prompt.input_ids.shape[1]] = -100

        with torch.no_grad():
            out = self.mdl(input_ids=input_ids, labels=labels)
        return float(out.loss.item())

    def _resolve_device(self, device: Optional[str]):
        if device and device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")



class OllamaClient(BaseModelClient):
    def __init__(self, model_id: str, api_base: Optional[str] = None):
        if requests is None:
            raise RuntimeError("Ollama backend requires requests installed.")
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
        return GenOutput(text=text)


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
