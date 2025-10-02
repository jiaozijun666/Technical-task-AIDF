# -*- coding: utf-8 -*-
"""
Unified evaluator for baseline methods.

- Public API: run_eval(...)
- Tries to call a baseline implementation from your `baseline/...` package first.
  The function signature we expect in those modules is:
      run(samples, tokenizer, model, build_prompt, cfg, limit) -> (auroc, outputs)
  where:
      samples: List[Dict] with keys {"question", "context", "gold"}
      tokenizer, model: HF objects (or None if not needed)
      build_prompt: callable(question:str, context:str|None, use_context:bool) -> str
      cfg: dict-like with fields {temperature, max_new_tokens, with_context, backend, api_base}
      limit: optional int to truncate the evaluation set
- If the target baseline cannot be imported, we fall back to a built-in
  "perplexity" implementation that works with the HF backend (and supports Ollama
  *generation only*, i.e., without NLL/perplexity scoring).

This file keeps *no* dataset- or paper-specific code. It just builds prompts,
runs generation/scoring, and writes JSONL.

Author: you :)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import argparse
import json
import math
import os
import sys
import time

# ---------- Optional imports (torch / transformers only when needed) ----------
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

# ---------- Prompt builder (prefer your existing src/prompt.py) ---------------
try:
    from src.prompt import build_prompt  # your own prompt builder if present
except Exception:
    def build_prompt(question: str, context: Optional[str], use_context: bool) -> str:
        """Tiny fallback prompt builder."""
        if use_context and context:
            return (
                "Use the CONTEXT to answer with a short phrase ONLY.\n"
                "Do not explain. Output just the answer.\n\n"
                f"CONTEXT: {context}\n\n"
                f"Question: {question}\nAnswer:"
            )
        return (
            "Answer the question with a short phrase ONLY.\n"
            "Do not explain. Output just the answer.\n\n"
            f"Question: {question}\nAnswer:"
        )

# ---------- Small I/O helpers -------------------------------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

# ---------- Simple normalization / metrics -----------------------------------
def _normalize_text(s: str) -> str:
    import re, string
    s = s.strip().lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s)
    return s

def exact_match(pred: str, gold: str) -> int:
    """Return 1 if exactly equal after normalization, else 0."""
    return int(_normalize_text(pred) == _normalize_text(gold))

def auroc_from_pairs(y_true: List[int], scores: List[float]) -> float:
    """
    Minimal AUROC implementation (no sklearn dependency).
    Positive class is y=1 (hallucination). Larger score = more likely positive.
    """
    # Sort by score descending
    pairs = sorted(zip(scores, y_true), key=lambda t: t[0], reverse=True)
    P = sum(y_true)
    N = len(y_true) - P
    if P == 0 or N == 0:
        return float("nan")
    tp = fp = 0
    prev_score = None
    auc = 0.0
    prev_tp = prev_fp = 0

    for score, y in pairs:
        if prev_score is None or score != prev_score:
            # trapezoid area between (prev_fp/N, prev_tp/P) and (fp/N, tp/P)
            auc += (fp - prev_fp) / N * (tp + prev_tp) / (2 * P)
            prev_score, prev_tp, prev_fp = score, tp, fp
        if y == 1:
            tp += 1
        else:
            fp += 1
    # last trapezoid
    auc += (fp - prev_fp) / N * (tp + prev_tp) / (2 * P)
    return float(auc)

# ---------- Backend: Ollama (HTTP) -------------------------------------------
def _ollama_generate(api_base: str, model: str, prompt: str, temperature: float, max_new_tokens: int) -> str:
    """
    Call Ollama's /api/generate endpoint. No loss/perplexity available here.
    """
    import requests  # lightweight dependency
    url = f"{api_base.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_new_tokens),
        },
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()

# ---------- Backend: HF (Transformers) ---------------------------------------
def _get_device() -> str:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    return "cpu"

def _hf_load(model_name: str):
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("transformers/torch not available. Please install requirements.")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name)
    device = _get_device()
    mdl.to(device)
    mdl.eval()
    return tok, mdl, device

@torch.inference_mode()
def _hf_generate_only(tokenizer, model, device: str, prompt: str, temperature: float, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # return only the continuation portion
    full = text
    # naive way to cut the prompt: if tokenizer is not reversible, we still return tail
    if full.startswith(prompt):
        return full[len(prompt):].strip()
    return full.strip()

@torch.inference_mode()
def _hf_per_token_nll(tokenizer, model, device: str, prompt: str, answer: str) -> float:
    """
    Compute mean negative log-likelihood on the *answer tokens* given the prompt.
    """
    full = prompt + (" " if (prompt and not prompt.endswith(" ")) else "") + answer
    enc = tokenizer(full, return_tensors="pt")
    enc_prompt = tokenizer(prompt, return_tensors="pt")

    input_ids = enc["input_ids"].to(device)
    labels = input_ids.clone()

    # mask prompt tokens to -100 so they don't contribute to the loss
    prompt_len = enc_prompt["input_ids"].shape[-1]
    labels[:, :prompt_len] = -100

    out = model(input_ids=input_ids, labels=labels)
    loss = out.loss  # mean over non-masked
    return float(loss.detach().cpu().item())

# ---------- Built-in baseline: perplexity ------------------------------------
def _builtin_perplexity(samples: List[Dict[str, Any]],
                        tokenizer,
                        model,
                        build_prompt_fn,
                        cfg: Dict[str, Any],
                        limit: Optional[int]) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Fallback perplexity baseline:
    - HF backend: generate + compute NLL on generated answer tokens
    - Ollama backend: generate only (scores unavailable -> AUROC likely NaN)
    """
    backend = cfg.get("backend", "hf")
    with_context = bool(cfg.get("with_context", False))
    temperature = float(cfg.get("temperature", 0.5))
    max_new_tokens = int(cfg.get("max_new_tokens", 16))
    api_base = cfg.get("api_base", "http://127.0.0.1:11434")

    rows: List[Dict[str, Any]] = []
    n = len(samples) if limit is None else min(len(samples), int(limit))

    y_true: List[int] = []
    scores: List[float] = []

    for i in range(n):
        ex = samples[i]
        q = ex["question"]
        ctx = ex.get("context")
        gold = ex["gold"]

        prompt = build_prompt_fn(q, ctx, with_context)

        if backend == "hf":
            assert tokenizer is not None and model is not None, "HF backend requires tokenizer/model."
            device = _get_device()
            pred = _hf_generate_only(tokenizer, model, device, prompt, temperature, max_new_tokens)
            try:
                nll = _hf_per_token_nll(tokenizer, model, device, prompt, pred)
            except Exception:
                nll = None
        elif backend == "ollama":
            pred = _ollama_generate(api_base, cfg["model"], prompt, temperature, max_new_tokens)
            nll = None
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # EM / hallucination label: 1 means hallucination (EM == 0)
        em = exact_match(pred, gold)
        label = 1 - em  # hallucination as positive class

        row = {
            "question": q,
            "context": ctx,
            "gold": gold,
            "pred": pred,
            "em": em,
            "loss": nll,
        }
        rows.append(row)

        if nll is not None and not math.isnan(nll) and math.isfinite(nll):
            y_true.append(label)
            scores.append(float(nll))

    # AUROC on available scores
    auroc = float("nan")
    if len(scores) >= 2 and len(set(y_true)) > 1:
        auroc = auroc_from_pairs(y_true, scores)

    return auroc, rows

# ---------- Registry: try to import your baseline implementations -------------
def _import_baseline_runner(name: str):
    """
    Try multiple canonical locations for a baseline runner.
    Expected callable signature:
        run(samples, tokenizer, model, build_prompt, cfg, limit) -> (auroc, outputs)
    """
    candidates = [
        f"baseline.uncertainty_based.{name}",
        f"baseline.internal_representation_based.{name}",
        f"baseline.{name}",
    ]
    for mod in candidates:
        try:
            m = __import__(mod, fromlist=["run"])
            if hasattr(m, "run") and callable(m.run):
                return m.run
        except Exception:
            continue
    return None

# ---------- Public API --------------------------------------------------------
@dataclass
class EvalConfig:
    baseline: str
    data: str
    out: str
    model: str
    backend: str = "hf"                  # "hf" or "ollama"
    temperature: float = 0.5
    max_new_tokens: int = 16
    with_context: bool = False
    limit: Optional[int] = None
    api_base: str = "http://127.0.0.1:11434"  # for Ollama

def run_eval(*,
             baseline: str,
             data: str,
             out: str,
             model: str,
             backend: str = "hf",
             temperature: float = 0.5,
             max_new_tokens: int = 16,
             with_context: bool = False,
             limit: Optional[int] = None,
             api_base: str = "http://127.0.0.1:11434") -> float:
    """
    Load data, run the specified baseline, save outputs, and return AUROC (float|NaN).
    """
    # 1) load
    samples = load_jsonl(data)

    # 2) load model if needed (HF backend); we pass tokenizer/model even if the
    #    imported runner does not need them.
    tokenizer = model_obj = device = None
    if backend == "hf":
        tokenizer, model_obj, device = _hf_load(model)

    # 3) baseline dispatch
    cfg = {
        "backend": backend,
        "model": model,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "with_context": with_context,
        "api_base": api_base,
    }

    runner = _import_baseline_runner(baseline)
    if runner is None:
        if baseline.lower() != "perplexity":
            raise RuntimeError(
                f"Baseline '{baseline}' not found in your 'baseline/' package "
                f"and no built-in implementation is available."
            )
        # fallback
        auroc, rows = _builtin_perplexity(samples, tokenizer, model_obj, build_prompt, cfg, limit)
    else:
        auroc, rows = runner(samples, tokenizer, model_obj, build_prompt, cfg, limit)

    # 4) save & report
    save_jsonl(out, rows)
    print(f"[generate/{baseline}] saved {len(rows)} rows to {out}")
    if auroc == auroc:  # not NaN
        print(f"[generate/{baseline}] AUROC = {auroc:.4f}")
    else:
        print(f"[generate/{baseline}] AUROC not computed (single class or invalid scores).")
    return float(auroc)

# ---------- Minimal CLI (optional) -------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="e.g., perplexity / p_true / mars / mars_se / semantic_entropy ...")
    p.add_argument("--data", required=True, help="Input JSONL file")
    p.add_argument("--out", required=True, help="Output JSONL file")
    p.add_argument("--model", required=True, help="HF model id or Ollama model name")
    p.add_argument("--backend", default="hf", choices=["hf", "ollama"])
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--with_context", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--api_base", default="http://127.0.0.1:11434", help="Ollama base URL")
    return p

def main() -> None:
    args = _build_argparser().parse_args()
    run_eval(
        baseline=args.baseline,
        data=args.data,
        out=args.out,
        model=args.model,
        backend=args.backend,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        with_context=args.with_context,
        limit=args.limit,
        api_base=args.api_base,
    )

if __name__ == "__main__":
    main()
