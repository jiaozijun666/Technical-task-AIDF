# baseline/uncertainty_based/mars.py
"""
MARS baseline (practical approximation).

We aggregate token-level confidence signals over the generation trajectory:
  - pmax_mean:   mean over steps of max softmax probability
  - margin_mean: mean over steps of (p_top1 - p_top2)
  - entropy_det: 1 - (mean entropy / log(V)), V=vocab size

Final confidence:
  conf = w_pmax * pmax_mean + w_margin * margin_mean + w_entropy * entropy_det
Hallucination score = 1 - conf  (higher => more likely hallucination)

AUROC is computed with hallucination (EM == 0) as the positive class.

Config (overridable on `cfg`):
  - mars_w_pmax: float = 0.5
  - mars_w_margin: float = 0.3
  - mars_w_entropy: float = 0.2
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import math
import torch
import torch.nn.functional as F

# absolute import (launch from project root)
from baseline.utils import exact_match, auroc_from_pairs


@torch.no_grad()
def _gen_with_scores(tokenizer, model, prompt: str, max_new_tokens: int, temperature: float):
    """
    Generate once and return:
      pred_text: str
      step_scores: List[Tensor [1, vocab]]  (logits for each generated step)
    """
    dev = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(dev)
    in_len = enc["input_ids"].shape[1]

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        output_scores=True,
        return_dict_in_generate=True,
    )
    pred_text = tokenizer.decode(out.sequences[0][in_len:], skip_special_tokens=True)
    return pred_text, out.scores  # scores: list of logits per step


def _trajectory_confidence(step_scores: List[torch.Tensor]) -> Dict[str, float]:
    """
    Compute pmax_mean, margin_mean, entropy_mean, entropy_det given a list
    of per-step logits tensors [1, vocab].
    """
    if not step_scores:
        return {"pmax_mean": 0.0, "margin_mean": 0.0, "entropy_mean": 0.0, "entropy_det": 0.0}

    pmax_list = []
    margin_list = []
    ent_list = []

    # Assume logits shape [1, V]
    V = step_scores[0].shape[-1]
    logV = math.log(V) if V and V > 1 else 1.0

    for logits in step_scores:
        probs = F.softmax(logits, dim=-1)[0]            # [V]
        # p_top1 and p_top2
        p_sorted, _ = torch.sort(probs, descending=True)
        p1 = float(p_sorted[0].item())
        p2 = float(p_sorted[1].item()) if p_sorted.shape[0] > 1 else 0.0

        # entropy
        ent = float(-(probs * (probs + 1e-12).log()).sum().item())

        pmax_list.append(p1)
        margin_list.append(max(0.0, p1 - p2))
        ent_list.append(ent)

    pmax_mean = float(sum(pmax_list) / len(pmax_list))
    margin_mean = float(sum(margin_list) / len(margin_list))
    entropy_mean = float(sum(ent_list) / len(ent_list))
    entropy_det = float(max(0.0, 1.0 - entropy_mean / max(logV, 1e-12)))  # normalized determinism

    return {
        "pmax_mean": pmax_mean,
        "margin_mean": margin_mean,
        "entropy_mean": entropy_mean,
        "entropy_det": entropy_det,
    }


def run(
    samples: List[Dict],
    tokenizer,
    model,
    build_prompt,   # (question: str, context: str | None, cfg) -> str
    cfg,            # needs: temperature, max_new_tokens; optional: mars_w_*
    limit: int | None = None,
) -> Tuple[float, List[Dict]]:
    """
    Execute the MARS baseline (approximation).
    Returns (auroc, outputs).
    """
    data = samples[:limit] if (limit and limit > 0) else samples

    # weights (can be overridden on cfg)
    w_pmax = float(getattr(cfg, "mars_w_pmax", 0.5))
    w_margin = float(getattr(cfg, "mars_w_margin", 0.3))
    w_entropy = float(getattr(cfg, "mars_w_entropy", 0.2))

    # normalize weights in case user sets arbitrary values
    w_sum = max(1e-12, (w_pmax + w_margin + w_entropy))
    w_pmax, w_margin, w_entropy = w_pmax / w_sum, w_margin / w_sum, w_entropy / w_sum

    y_true: List[int] = []     # 1 = hallucination, 0 = correct
    y_score: List[float] = []  # hallucination score = 1 - confidence
    outputs: List[Dict] = []

    for ex in data:
        q = ex["question"]
        ctx = ex.get("context", "")

        prompt = build_prompt(q, ctx, cfg)
        pred, step_scores = _gen_with_scores(
            tokenizer, model, prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )

        feats = _trajectory_confidence(step_scores)
        conf = (
            w_pmax * feats["pmax_mean"] +
            w_margin * feats["margin_mean"] +
            w_entropy * feats["entropy_det"]
        )
        h_score = float(max(0.0, min(1.0, 1.0 - conf)))  # clamp to [0,1]

        em = exact_match(pred, ex["gold"])  # 1 = correct, 0 = hallucination
        y_true.append(1 - em)
        y_score.append(h_score)

        outputs.append({
            "question": q,
            "gold": ex["gold"],
            "pred": pred.strip(),
            "em_correct": int(em),
            "mars_confidence": float(conf),
            "pmax_mean": float(feats["pmax_mean"]),
            "margin_mean": float(feats["margin_mean"]),
            "entropy_mean": float(feats["entropy_mean"]),
            "entropy_det": float(feats["entropy_det"]),
            "halluc_score": h_score,
        })

    auroc = auroc_from_pairs(y_true, y_score)
    return auroc, outputs
