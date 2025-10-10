import numpy as np
import torch
from torch.nn import functional as F

def compute_perplexity(model, tokenizer, prompt, answer):
    """Compute perplexity = exp(mean NLL)."""
    inputs = tokenizer(prompt + answer, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        return torch.exp(outputs.loss).item()

def evaluate(model, tokenizer, dataset):
    """
    Baseline: Perplexity
    Uses model negative log-likelihood as confidence measure.
    """
    results = []
    for item in dataset:
        q = item.get("question", "")
        pos = item.get("gold") or item.get("pos")
        neg = item.get("neg") or item.get("answer2")
        if not pos or not neg:
            continue
        try:
            ppl_true = compute_perplexity(model, tokenizer, q, pos)
            ppl_false = compute_perplexity(model, tokenizer, q, neg)
        except Exception as e:
            print(f"[Perplexity] Error: {e}")
            continue
        # lower perplexity means higher confidence → flip sign for consistency
        results.append({"label": 1, "score": -ppl_true})
        results.append({"label": 0, "score": -ppl_false})
    return results