"""
HaloScope-like baseline (unsupervised membership estimation, practical).

Steps:
  - For each question, generate once with logits and extract features:
      f = [pmax_mean, margin_mean, entropy_det]
  - Fit a 2-component Gaussian Mixture on all features (unsupervised).
  - Choose the "correct" component as the one with larger mean of
    (+pmax_mean, +margin_mean, +entropy_det).
  - Hallucination score = posterior probability of the "incorrect" component.

Positive class for AUROC = hallucination (EM == 0).
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
from sklearn.mixture import GaussianMixture

from baseline.utils import exact_match, auroc_from_pairs
from baseline.uncertainty_based.mars import _gen_with_scores, _trajectory_confidence


def run(
    samples: List[Dict],
    tokenizer,
    model,
    build_prompt,   # (question, context, cfg) -> str
    cfg,
    limit: int | None = None,
) -> Tuple[float, List[Dict]]:
    data = samples[:limit] if (limit and limit > 0) else samples

    feats_list = []
    cache = []  # keep per-sample info to avoid re-generation
    for ex in data:
        q = ex["question"]; ctx = ex.get("context", "")
        prompt = build_prompt(q, ctx, cfg)
        pred, step_scores = _gen_with_scores(
            tokenizer, model, prompt,
            max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature
        )
        feats = _trajectory_confidence(step_scores)
        x = np.array([feats["pmax_mean"], feats["margin_mean"], feats["entropy_det"]], dtype=np.float32)
        feats_list.append(x)
        cache.append((ex, pred, feats, x))

    X = np.stack(feats_list, axis=0) if feats_list else np.zeros((0, 3), dtype=np.float32)
    if len(X) >= 2:
        gm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
        gm.fit(X)
        post = gm.predict_proba(X)  # [N, 2]
        # decide which component is "correct": larger mean of (pmax, margin, entropy_det)
        means = gm.means_  # [2, 3]
        score_means = means @ np.array([1.0, 1.0, 1.0])  # simple additive criterion
        correct_comp = int(score_means.argmax())
        incorrect_comp = 1 - correct_comp
        hallu = post[:, incorrect_comp]  # hallucination score
    else:
        hallu = np.zeros((len(X),), dtype=np.float32)

    y_true, y_score, outputs = [], [], []
    for i, (ex, pred, feats, _) in enumerate(cache):
        em = exact_match(pred, ex["gold"])
        y_true.append(1 - em)
        h = float(hallu[i]) if len(hallu) > i else 0.0
        y_score.append(h)
        outputs.append({
            "question": ex["question"], "gold": ex["gold"], "pred": pred.strip(),
            "em_correct": int(em),
            "pmax_mean": float(feats["pmax_mean"]),
            "margin_mean": float(feats["margin_mean"]),
            "entropy_det": float(feats["entropy_det"]),
            "halluc_score": h,
        })

    auroc = auroc_from_pairs(y_true, y_score)
    return auroc, outputs
