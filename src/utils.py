# src/utils.py
from __future__ import annotations
import re, string
from typing import Sequence
import numpy as np
import torch
import torch.nn.functional as F

def normalize_text(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def exact_match(pred: str, golds: Sequence[str]) -> int:
    p = normalize_text(pred)
    for g in golds:
        if p == normalize_text(g):
            return 1
    return 0

def softmax(x):
    x = np.asarray(x, dtype=float)
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()

def compute_logprob(model, tok, prompt: str, completion: str) -> float:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    labels = tok(completion, return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad():
        out = model(**inputs, labels=labels)
        loss = out.loss
    return float(-loss.detach().to(torch.float32).item())

def agreement(model, tok, q: str, a1: str, a2: str) -> float:
    p1 = np.exp(compute_logprob(model, tok, q, a1))
    p2 = np.exp(compute_logprob(model, tok, q, a2))
    return float(p1 / (p1 + p2 + 1e-9))

def temperature_scaled_agreement(model, tok, q: str, a1: str, a2: str, T: float = 1.5) -> float:
    lp1 = compute_logprob(model, tok, q, a1)
    lp2 = compute_logprob(model, tok, q, a2)
    z = (lp1 - lp2) / max(T, 1e-6)
    return float(1.0 / (1.0 + np.exp(-z)))

__all__ = [
    "normalize_text",
    "exact_match",
    "softmax",
    "compute_logprob",
    "agreement",
    "temperature_scaled_agreement",
]
