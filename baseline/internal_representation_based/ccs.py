"""
CCS (Contrast Consistency) baseline — practical approximation.

For each question:
  1) Build the same prompt twice but with slight contrast (temperature/style).
  2) Generate two trajectories with logits (output_scores=True).
  3) Align by step index; compute:
       - mean symmetric KL between per-step softmax distributions
       - top-1 token agreement ratio across aligned steps
  4) Consistency score:
       cons = exp(-mean_sym_kl) * (beta + (1 - beta) * top1_agree)
     Hallucination score = 1 - cons

Config on `cfg` (with defaults):
  - ccs_delta_temp: float = 0.4
  - ccs_beta: float = 0.5          # weight for multiplicative fusion with top1_agree
  - temperature, max_new_tokens    # from your PromptConfig
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import math
import torch
import torch.nn.functional as F

from baseline.utils import exact_match, auroc_from_pairs
from baseline.uncertainty_based.mars import _gen_with_scores  # reuse generator


def _sym_kl(p: torch.Tensor, q: torch.Tensor) -> float:
    """Symmetric KL for probabilities p, q (1D)."""
    eps = 1e-12
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    kl_pq = (p * (p / q).log()).sum()
    kl_qp = (q * (q / p).log()).sum()
    return float((kl_pq + kl_qp).item())


def _contrast_consistency(step_scores_a: List[torch.Tensor],
                          step_scores_b: List[torch.Tensor]) -> Tuple[float, float]:
    """
    Return (mean_sym_kl, top1_agree) given two sequences of per-step logits.
    """
    T = min(len(step_scores_a), len(step_scores_b))
    if T == 0:
        return 0.0, 0.0

    sym_kls = []
    agree = 0
    for t in range(T):
        pa = F.softmax(step_scores_a[t][0], dim=-1)
        pb = F.softmax(step_scores_b[t][0], dim=-1)
        sym_kls.append(_sym_kl(pa, pb))
        if int(pa.argmax()) == int(pb.argmax()):
            agree += 1

    mean_sym_kl = sum(sym_kls) / len(sym_kls)
    top1_agree = agree / T
    return float(mean_sym_kl), float(top1_agree)


def run(
    samples: List[Dict],
    tokenizer,
    model,
    build_prompt,   # (question, context, cfg) -> str
    cfg,
    limit: int | None = None,
) -> Tuple[float, List[Dict]]:
    """
    Execute CCS baseline. Returns (auroc, outputs). Positive class = hallucination (EM == 0).
    """
    data = samples[:limit] if (limit and limit > 0) else samples

    delta_temp = float(getattr(cfg, "ccs_delta_temp", 0.4))
    beta = float(getattr(cfg, "ccs_beta", 0.5))

    y_true, y_score, outputs = [], [], []

    for ex in data:
        q = ex["question"]; ctx = ex.get("context", "")
        prompt = build_prompt(q, ctx, cfg)

        # run A (base temperature)
        pred_a, scores_a = _gen_with_scores(
            tokenizer, model, prompt,
            max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature
        )
        # run B (contrasted temperature)
        pred_b, scores_b = _gen_with_scores(
            tokenizer, model, prompt,
            max_new_tokens=cfg.max_new_tokens, temperature=max(0.0, cfg.temperature + delta_temp)
        )

        mean_sym_kl, top1_agree = _contrast_consistency(scores_a, scores_b)
        cons = math.exp(-mean_sym_kl) * (beta + (1.0 - beta) * top1_agree)
        cons = max(0.0, min(1.0, cons))
        h_score = 1.0 - cons

        # use run A text as prediction for EM
        em = exact_match(pred_a, ex["gold"])
        y_true.append(1 - em)
        y_score.append(h_score)

        outputs.append({
            "question": q, "gold": ex["gold"],
            "pred": pred_a.strip(),
            "em_correct": int(em),
            "ccs_mean_sym_kl": float(mean_sym_kl),
            "ccs_top1_agree": float(top1_agree),
            "ccs_consistency": float(cons),
            "halluc_score": float(h_score),
        })

    auroc = auroc_from_pairs(y_true, y_score)
    return auroc, outputs
