"""
SAPLMA-like baseline (supervised small MLP on internal features).

Pipeline:
  - For each sample, generate once and extract features:
      x = [pmax_mean, margin_mean, entropy_det]
      y = 1 if EM-correct else 0
  - Use first K samples as training (auto-labeled by EM), train a tiny MLP
    (or logistic regression) to predict correctness probability.
  - Hallucination score = 1 - P(correct | x)

Config on `cfg` (defaults):
  - saplma_train_k: int = 100
  - saplma_hidden: int = 64
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np

from sklearn.neural_network import MLPClassifier
# from sklearn.linear_model import LogisticRegression  # alternative

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

    train_k = int(getattr(cfg, "saplma_train_k", 100))
    hidden = int(getattr(cfg, "saplma_hidden", 64))

    X, Y, cache = [], [], []
    for ex in data:
        q = ex["question"]; ctx = ex.get("context", "")
        prompt = build_prompt(q, ctx, cfg)
        pred, step_scores = _gen_with_scores(
            tokenizer, model, prompt,
            max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature
        )
        feats = _trajectory_confidence(step_scores)
        x = np.array([feats["pmax_mean"], feats["margin_mean"], feats["entropy_det"]], dtype=np.float32)
        y = int(exact_match(pred, ex["gold"]))  # 1=correct, 0=incorrect
        X.append(x); Y.append(y)
        cache.append((ex, pred, feats, x, y))

    if len(X) == 0:
        return 0.0, []

    X = np.stack(X, axis=0); Y = np.array(Y, dtype=np.int32)

    k = min(train_k, len(X) // 2 if len(X) >= 2 else 1)
    Xtr, Ytr = X[:k], Y[:k]
    Xte, Yte = X[k:], Y[k:]

    # A tiny MLP; you can switch to LogisticRegression if preferred.
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), activation="relu",
                        max_iter=200, random_state=0)
    # LogisticRegression alternative:
    # clf = LogisticRegression(max_iter=200)

    # Handle the case where Ytr may be all-1 or all-0 (ill-conditioned)
    if len(np.unique(Ytr)) == 1:
        # Fall back to a dummy thresholding rule on pmax_mean
        thresh = float(np.median(Xtr[:, 0]))
        proba_te = 1.0 / (1.0 + np.exp(-10.0 * (Xte[:, 0] - thresh)))  # ~sigmoid
        prob_correct_te = proba_te if Ytr[0] == 1 else (1.0 - proba_te)
        prob_correct = np.concatenate([np.ones(k) * Ytr[0], prob_correct_te], axis=0)
    else:
        clf.fit(Xtr, Ytr)
        prob_correct = np.zeros((len(X),), dtype=np.float32)
        prob_correct[:k] = clf.predict_proba(Xtr)[:, 1]
        prob_correct[k:] = clf.predict_proba(Xte)[:, 1]

    # hallucination score = 1 - P(correct)
    hallu = 1.0 - prob_correct

    y_true, y_score, outputs = [], [], []
    for i, (ex, pred, feats, _, y) in enumerate(cache):
        y_true.append(1 - y)  # positive = hallucination
        h = float(hallu[i])
        y_score.append(h)
        outputs.append({
            "question": ex["question"], "gold": ex["gold"], "pred": pred.strip(),
            "em_correct": int(y),
            "pmax_mean": float(feats["pmax_mean"]),
            "margin_mean": float(feats["margin_mean"]),
            "entropy_det": float(feats["entropy_det"]),
            "saplma_prob_correct": float(1.0 - h),
            "halluc_score": h,
            "is_train": int(i < k),
        })

    auroc = auroc_from_pairs(y_true, y_score)
    return auroc, outputs
