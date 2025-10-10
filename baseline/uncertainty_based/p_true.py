import numpy as np
import torch
from torch.nn import functional as F

def compute_nll(model, tokenizer, prompt, answer):
    """Compute mean negative log-likelihood as in HaMI Eq.(6)."""
    inputs = tokenizer(prompt + answer, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        # outputs.loss = mean cross-entropy = mean NLL
        return outputs.loss.item()

def evaluate(model, tokenizer, dataset):
    """
    Baseline: p(True)
    Computes the log-probability (NLL) of the model predicting the correct answer.
    """
    results = []
    for item in dataset:
        q = item.get("question", "")
        pos = item.get("gold") or item.get("pos")
        neg = item.get("neg") or item.get("answer2")
        if not pos or not neg:
            continue
        try:
            p_true = compute_nll(model, tokenizer, q, pos)
            p_false = compute_nll(model, tokenizer, q, neg)
        except Exception as e:
            print(f"[p_true] Error: {e}")
            continue
        results.append({"label": 1, "score": -p_true})   # higher = more likely correct
        results.append({"label": 0, "score": -p_false})
    return results


