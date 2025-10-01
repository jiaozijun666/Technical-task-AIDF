# baseline/uncertainty_based/mars_se.py
"""
MARS-SE baseline (fusion of trajectory confidence and semantic agreement).

Pipeline per sample:
  1) Build prompt and generate once to obtain token-level logits sequence.
  2) MARS confidence:
       - from logits, compute pmax_mean / margin_mean / entropy_det (see mars.py)
       - combine them into conf_MARS in [0,1]
  3) SE agreement:
       - sample K answers for the same prompt
       - measure mutual entailment ratio across pairs (GPT if enabled,
         otherwise MiniLM cosine fallback), yielding agreement in [0,1]
  4) Fuse into a hallucination score:
       h = alpha * (1 - conf_MARS) + (1 - alpha) * (1 - agreement)

Positive class for AUROC is hallucination (EM == 0).

Config knobs on `cfg` (with defaults):
  - mars_w_pmax: float = 0.5      # weights used by mars.py to form conf_MARS
  - mars_w_margin: float = 0.3
  - mars_w_entropy: float = 0.2
  - mars_se_alpha: float = 0.6    # fusion weight alpha

  - se_k: int = 5                 # number of samples for SE
  - se_use_gpt: bool = True       # prefer GPT entailment if available
  - se_gpt_model: str = "gpt-3.5-turbo"
  - se_sim_threshold: float = 0.80  # MiniLM cosine threshold fallback
"""

from __future__ import annotations
from typing import List, Dict, Tuple

import torch

# absolute imports to avoid relative-import pitfalls; run from project root
from baseline.utils import exact_match, auroc_from_pairs
from baseline.uncertainty_based.mars import _gen_with_scores, _trajectory_confidence
from baseline.uncertainty_based.semantic_entropy import (
    _sample_answers,
    _agreement_across_answers,
)


def run(
    samples: List[Dict],
    tokenizer,
    model,
    build_prompt,   # (question: str, context: str | None, cfg) -> str
    cfg,            # needs: temperature, max_new_tokens; plus cfg knobs above
    limit: int | None = None,
) -> Tuple[float, List[Dict]]:
    """
    Execute the MARS-SE fusion baseline. Returns (auroc, outputs).
    """
    data = samples[:limit] if (limit and limit > 0) else samples

    # Fusion weight
    alpha = float(getattr(cfg, "mars_se_alpha", 0.6))

    # SE knobs
    K = int(getattr(cfg, "se_k", 5))
    USE_GPT = bool(getattr(cfg, "se_use_gpt", True))
    GPT_MODEL = getattr(cfg, "se_gpt_model", "gpt-3.5-turbo")
    SIM_THR = float(getattr(cfg, "se_sim_threshold", 0.80))

    y_true: List[int] = []      # 1 = hallucination, 0 = correct
    y_score: List[float] = []   # fused hallucination score
    outputs: List[Dict] = []

    for ex in data:
        q = ex["question"]
        ctx = ex.get("context", "")

        # ----- Build prompt -----
        prompt = build_prompt(q, ctx, cfg)

        # ----- MARS confidence (re-using mars.py helpers) -----
        pred, step_scores = _gen_with_scores(
            tokenizer, model, prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )
        feats = _trajectory_confidence(step_scores)
        # mars.py already normalizes its own feature ranges; the weights it uses are
        # read from cfg internally there. We recompute conf here to avoid importing
        # internal weight logic; honor the same knobs on cfg.
        w_p = float(getattr(cfg, "mars_w_pmax", 0.5))
        w_m = float(getattr(cfg, "mars_w_margin", 0.3))
        w_e = float(getattr(cfg, "mars_w_entropy", 0.2))
        s = max(1e-12, w_p + w_m + w_e)
        w_p, w_m, w_e = w_p / s, w_m / s, w_e / s

        conf_mars = (
            w_p * feats["pmax_mean"] +
            w_m * feats["margin_mean"] +
            w_e * feats["entropy_det"]
        )
        conf_mars = float(max(0.0, min(1.0, conf_mars)))  # clamp

        # ----- SE agreement (re-using semantic_entropy helpers) -----
        answers = _sample_answers(
            tokenizer, model, prompt,
            k=K, max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature
        )
        agreement = _agreement_across_answers(
            answers, use_gpt=USE_GPT, gpt_model=GPT_MODEL, sim_threshold=SIM_THR
        )
        agreement = float(max(0.0, min(1.0, agreement)))

        # ----- Fusion into hallucination score -----
        h_mars = 1.0 - conf_mars
        h_se = 1.0 - agreement
        h_score = float(max(0.0, min(1.0, alpha * h_mars + (1.0 - alpha) * h_se)))

        # ----- Label and logs -----
        em = exact_match(pred, ex["gold"])  # 1=correct
        y_true.append(1 - em)               # hallucination as positive class
        y_score.append(h_score)

        outputs.append({
            "question": q,
            "gold": ex["gold"],
            "pred": pred.strip(),
            "em_correct": int(em),

            # MARS parts
            "mars_confidence": conf_mars,
            "pmax_mean": float(feats["pmax_mean"]),
            "margin_mean": float(feats["margin_mean"]),
            "entropy_det": float(feats["entropy_det"]),

            # SE parts
            "agreement": agreement,
            "answers": answers,  # keep for inspection; remove if file size matters

            # Fusion
            "alpha": alpha,
            "halluc_score": h_score,
        })

    auroc = auroc_from_pairs(y_true, y_score)
    return auroc, outputs
