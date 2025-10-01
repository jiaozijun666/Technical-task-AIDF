from __future__ import annotations
from typing import List, Dict, Tuple
import torch
from sklearn.metrics import roc_auc_score

def normalize(text: str) -> str:
    import re
    s = text.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()

def exact_match(pred: str, gold_list: List[str]) -> int:
    p = normalize(pred)
    gold = [normalize(g) for g in gold_list]
    return int(any(p == g for g in gold))  # 1 = correct, 0 = hallucination

@torch.no_grad()
def generate_and_get_scores(tokenizer, model, prompt: str, max_new_tokens: int, temperature: float) -> Tuple[str, float, list]:
    """
    Generate continuation and return:
        pred_text: str
        perpl_score: mean(-log p_max) over generated tokens
        step_scores: list of per-step logits (for advanced baselines)
    """
    dev = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(dev)
    in_len = inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        output_scores=True,
        return_dict_in_generate=True,
    )

    pred_text = tokenizer.decode(out.sequences[0][in_len:], skip_special_tokens=True)

    import torch as T
    scores = T.stack(out.scores, dim=0)    # [T, vocab]
    probs = scores.softmax(-1)
    p_max, _ = probs.max(-1)
    perpl = (-p_max.log()).mean().item()

    return pred_text, perpl, out.scores

def auroc_from_pairs(y_true: List[int], y_score: List[float]) -> float:
    if len(set(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)
