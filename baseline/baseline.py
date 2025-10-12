# baseline.py
# Unified baselines for hallucination detection:
# p(True), Perplexity, SE, MARS, MARS-SE, CCS, SAPLMA, HaloScope
# Author: you :)
# Dependencies: numpy (required), scipy (optional for DBSCAN). No torch required here.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any, Tuple
import math
import numpy as np

try:
    from scipy.spatial.distance import cdist
    from scipy.cluster.hierarchy import fclusterdata
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ---------------------------
# Data containers
# ---------------------------

@dataclass
class Generation:
    """One free-form generation of the same question."""
    text: str
    tokens: List[str]
    # token-level next-token log-probabilities (log p(y_t | y_<t, x))
    logprobs: List[float]
    # hidden states for the *generated* tokens (shape: T x D)
    hidden: Optional[np.ndarray] = None
    # optional: model-reported self probability (for p(True) style prompting)
    self_prob_true: Optional[float] = None


@dataclass
class Sample:
    """A training/test sample centered on one question."""
    question: str
    gold: Optional[str] = None
    # the primary generation we want to score
    gen: Generation = None
    # multiple generations for multi-sample baselines (SE/MARS-SE)
    gens: List[Generation] = field(default_factory=list)
    # Optional known correctness label for supervised probes
    label: Optional[int] = None  # 1 = hallucinated (positive), 0 = correct (negative)


# ---------------------------
# Utilities
# ---------------------------

def _safe_mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if len(xs) else 0.0

def _length_norm_nll(logprobs: List[float]) -> float:
    """Average NLL (negative mean log prob)."""
    if not logprobs:
        return 0.0
    return float(-np.mean(np.array(logprobs)))

def _perplexity(logprobs: List[float]) -> float:
    """Per-token perplexity = exp(average NLL)."""
    return math.exp(_length_norm_nll(logprobs))

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return num / den

def _pairwise_cosine(e: np.ndarray) -> np.ndarray:
    # e: M x d
    n = e.shape[0]
    sims = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            s = _cosine_sim(e[i], e[j])
            sims[i, j] = sims[j, i] = s
    return sims

def _simple_embed(
    texts: List[str],
    embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
) -> np.ndarray:
    """
    Returns M x d embeddings. If embed_fn is None, uses a TF-IDF-ish bag-of-char fallback,
    which is weak but keeps pipeline runnable.
    """
    if embed_fn is not None:
        return embed_fn(texts)

    # Fallback: cheap hashed character n-gram features
    dim = 512
    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for ch in t.lower():
            idx = (ord(ch) * 1315423911) % dim
            vecs[i, idx] += 1.0
        # L2 normalize
        nrm = np.linalg.norm(vecs[i]) + 1e-9
        vecs[i] /= nrm
    return vecs


def _cluster_entropy_from_embeds(E: np.ndarray, thresh: float = 0.25) -> Tuple[float, Dict[int, int]]:
    """
    Approximate Semantic Entropy (SE):
    - cluster M generations in embedding space
    - compute cluster distribution p(c) and entropy H = -sum p log p
    Returns (entropy, cluster_counts)
    """
    M = E.shape[0]
    if M <= 1:
        return 0.0, {0: M}

    # Simple agglomerative-style clustering via threshold on cosine distance
    # Build pairwise cosine similarity
    sims = _pairwise_cosine(E)
    # Simple union-find by greedy linkage on similarity >= (1 - thresh)
    parent = list(range(M))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(M):
        for j in range(i+1, M):
            if (1.0 - sims[i, j]) <= thresh:
                union(i, j)

    # compress and count
    for i in range(M):
        parent[i] = find(i)
    counts: Dict[int, int] = {}
    for p in parent:
        counts[p] = counts.get(p, 0) + 1

    probs = np.array(list(counts.values()), dtype=np.float32) / float(M)
    entropy = float(-np.sum(probs * (np.log(probs + 1e-12))))
    return entropy, counts


def _minmax_scale(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# ---------------------------
# Base API
# ---------------------------

class BaseDetector:
    name: str = "Base"

    def fit(self, samples: List[Sample]) -> "BaseDetector":
        return self

    def predict(self, s: Sample) -> float:
        """Return a scalar 'hallucination score', higher = more likely hallucination."""
        raise NotImplementedError

    def predict_many(self, samples: List[Sample]) -> np.ndarray:
        return np.array([self.predict(s) for s in samples], dtype=np.float32)


# ---------------------------
# 1) p(True)
# ---------------------------

class PTrueDetector(BaseDetector):
    """
    Prompts the model to self-report probability that the answer is true.
    You must pass a callable: get_p_true(question, answer)->float in [0,1].
    Score is 1 - p_true (higher => more hallucination).
    """
    name = "p(True)"

    def __init__(self, get_p_true: Optional[Callable[[str, str], float]] = None):
        self.get_p_true = get_p_true

    def predict(self, s: Sample) -> float:
        if s.gen.self_prob_true is not None:
            p_true = float(s.gen.self_prob_true)
        elif self.get_p_true is not None:
            p_true = float(self.get_p_true(s.question, s.gen.text))
        else:
            # fallback: monotonic map from perplexity -> pseudo p_true
            ppx = _perplexity(s.gen.logprobs)
            p_true = 1.0 / (1.0 + math.log1p(ppx))
        p_true = max(0.0, min(1.0, p_true))
        return 1.0 - p_true


# ---------------------------
# 2) Perplexity
# ---------------------------

class PerplexityDetector(BaseDetector):
    """Score = normalized perplexity (higher = more hallucination)."""
    name = "Perplexity"

    def predict(self, s: Sample) -> float:
        return float(_perplexity(s.gen.logprobs))


# ---------------------------
# 3) Semantic Entropy (SE)
# ---------------------------

class SEDetector(BaseDetector):
    """
    Approximates Semantic Entropy by clustering multiple generations in embedding space.
    Provide embed_fn: List[str] -> np.ndarray (M x d) for best results.
    """
    name = "SE"

    def __init__(self, embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None, dist_thresh: float = 0.25):
        self.embed_fn = embed_fn
        self.dist_thresh = dist_thresh

    def predict(self, s: Sample) -> float:
        gens = s.gens if s.gens else [s.gen]
        texts = [g.text for g in gens]
        E = _simple_embed(texts, self.embed_fn)
        entropy, _ = _cluster_entropy_from_embeds(E, thresh=self.dist_thresh)
        # Entropy is already "higher = more disagreement/instability" -> more hallucination.
        return float(entropy)


# ---------------------------
# 4) MARS (Meaning-aware Response Scoring)
# ---------------------------

class MARSDetector(BaseDetector):
    """
    A pragmatic approximation:
      - token-level surprise = -log p_t
      - sentence-level weight = semantic centrality of this generation among peers
      - score = centrality * avg surprise
    For a faithful reproduction, replace `semantic_centrality` with the paper’s method.
    """
    name = "MARS"

    def __init__(self, embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None):
        self.embed_fn = embed_fn

    def _centrality(self, s: Sample) -> float:
        if not s.gens:
            return 1.0
        texts = [g.text for g in s.gens]
        E = _simple_embed(texts, self.embed_fn)
        sims = _pairwise_cosine(E)
        # centrality of the primary generation relative to the bag
        idx0 = 0  # assume s.gen is first in s.gens if present; else use own text appended
        cent = float(np.mean(sims[idx0]))
        # map from [-1,1] to [0,1]
        return (cent + 1.0) / 2.0

    def predict(self, s: Sample) -> float:
        nll = _length_norm_nll(s.gen.logprobs)
        cent = self._centrality(s)
        # score higher = more hallucination => use cent-weighted NLL
        return float(cent * nll)


# ---------------------------
# 5) MARS-SE (MARS + SE)
# ---------------------------

class MARSSEDetector(BaseDetector):
    name = "MARS-SE"

    def __init__(self, embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None, se_weight: float = 1.0):
        self.mars = MARSDetector(embed_fn=embed_fn)
        self.se = SEDetector(embed_fn=embed_fn)
        self.se_weight = se_weight

    def predict(self, s: Sample) -> float:
        mars = self.mars.predict(s)
        se = self.se.predict(s)
        return float(mars + self.se_weight * se)


# ---------------------------
# 6) CCS (linear probe on internal states)
# ---------------------------

class CCSDetector(BaseDetector):
    """
    Simple logistic regression probe on last-token hidden states.
    You can pass which token index to use via token_selector; default uses last token.
    """
    name = "CCS"

    def __init__(self, token_selector: Optional[Callable[[Generation], int]] = None, lr=0.1, epochs=200, reg=1e-4):
        self.token_selector = token_selector or (lambda g: len(g.tokens) - 1)
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.lr = lr
        self.epochs = epochs
        self.reg = reg

    def fit(self, samples: List[Sample]) -> "CCSDetector":
        X, y = [], []
        for s in samples:
            if s.label is None or s.gen.hidden is None:
                continue
            idx = self.token_selector(s.gen)
            h = s.gen.hidden[idx]  # D
            X.append(h)
            y.append(int(s.label))
        if not X:
            return self

        X = np.stack(X, axis=0).astype(np.float32)  # N x D
        y = np.array(y, dtype=np.float32)  # N

        # init
        N, D = X.shape
        self.w = np.zeros(D, dtype=np.float32)
        self.b = 0.0

        # simple gradient descent
        for _ in range(self.epochs):
            logits = X @ self.w + self.b  # N
            probs = 1.0 / (1.0 + np.exp(-logits))
            grad_w = (X.T @ (probs - y)) / N + self.reg * self.w
            grad_b = float(np.mean(probs - y))
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict(self, s: Sample) -> float:
        if self.w is None or s.gen.hidden is None or len(s.gen.hidden) == 0:
            # fallback: use perplexity
            return float(_perplexity(s.gen.logprobs))
        idx = self.token_selector(s.gen)
        h = s.gen.hidden[idx]
        z = float(np.dot(h, self.w) + self.b)
        p = 1.0 / (1.0 + math.exp(-z))
        return float(p)  # interpret as hallucination probability


# ---------------------------
# 7) SAPLMA (MLP probe with labels)
# ---------------------------

class SAPLMADetector(BaseDetector):
    """
    Small two-layer MLP with ReLU on last-token hidden states.
    """
    name = "SAPLMA"

    def __init__(self, token_selector: Optional[Callable[[Generation], int]] = None,
                 hidden_dim: int = 256, lr: float = 1e-2, epochs: int = 300, reg: float = 1e-4):
        self.token_selector = token_selector or (lambda g: len(g.tokens) - 1)
        self.W1: Optional[np.ndarray] = None
        self.b1: Optional[np.ndarray] = None
        self.W2: Optional[np.ndarray] = None
        self.b2: float = 0.0
        self.hdim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.reg = reg

    def fit(self, samples: List[Sample]) -> "SAPLMADetector":
        X, y = [], []
        for s in samples:
            if s.label is None or s.gen.hidden is None:
                continue
            idx = self.token_selector(s.gen)
            X.append(s.gen.hidden[idx])
            y.append(int(s.label))
        if not X:
            return self

        X = np.stack(X, axis=0).astype(np.float32)
        y = np.array(y, dtype=np.float32)

        N, D = X.shape
        rng = np.random.default_rng(0)
        self.W1 = (rng.standard_normal((D, self.hdim)).astype(np.float32) / math.sqrt(D))
        self.b1 = np.zeros(self.hdim, dtype=np.float32)
        self.W2 = (rng.standard_normal((self.hdim, 1)).astype(np.float32) / math.sqrt(self.hdim))
        self.b2 = 0.0

        for _ in range(self.epochs):
            # forward
            h1 = np.maximum(0.0, X @ self.W1 + self.b1)     # N x H
            logits = (h1 @ self.W2).reshape(-1) + self.b2    # N
            probs = 1.0 / (1.0 + np.exp(-logits))
            # loss: BCE + L2
            # grad
            dlogits = (probs - y) / N                        # N
            dW2 = h1.T @ dlogits.reshape(-1, 1) + self.reg * self.W2
            db2 = float(np.sum(dlogits))
            dh1 = dlogits.reshape(-1, 1) @ self.W2.T         # N x H
            dh1[h1 <= 1e-12] = 0.0
            dW1 = X.T @ dh1 + self.reg * self.W1
            db1 = np.sum(dh1, axis=0)

            # update
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
        return self

    def predict(self, s: Sample) -> float:
        if self.W1 is None or s.gen.hidden is None or len(s.gen.hidden) == 0:
            return float(_perplexity(s.gen.logprobs))
        idx = self.token_selector(s.gen)
        x = s.gen.hidden[idx]
        h1 = np.maximum(0.0, x @ self.W1 + self.b1)
        z = float(h1 @ self.W2.reshape(-1) + self.b2)
        p = 1.0 / (1.0 + math.exp(-z))
        return float(p)


# ---------------------------
# 8) HaloScope (unsupervised membership score -> label/score)
# ---------------------------

class HaloScopeDetector(BaseDetector):
    """
    Lightweight surrogate of HaloScope idea:
      - learn a reference distribution of hidden states from *unlabeled* generations
      - score = Mahalanobis distance of last-token hidden to the reference (higher => more 'unusual')
    You can call fit() without labels; pass many samples to learn the reference stats.
    """
    name = "HaloScope"

    def __init__(self, token_selector: Optional[Callable[[Generation], int]] = None, eps: float = 1e-3):
        self.token_selector = token_selector or (lambda g: len(g.tokens) - 1)
        self.mu: Optional[np.ndarray] = None
        self.Sinv: Optional[np.ndarray] = None
        self.eps = eps

    def fit(self, samples: List[Sample]) -> "HaloScopeDetector":
        X = []
        for s in samples:
            if s.gen.hidden is None or len(s.gen.hidden) == 0:
                continue
            idx = self.token_selector(s.gen)
            X.append(s.gen.hidden[idx])
        if not X:
            return self
        X = np.stack(X, axis=0).astype(np.float32)
        self.mu = np.mean(X, axis=0)
        # shrinkage covariance inverse
        Xc = X - self.mu
        cov = (Xc.T @ Xc) / max(1, (X.shape[0] - 1))
        cov += self.eps * np.eye(cov.shape[0], dtype=np.float32)
        self.Sinv = np.linalg.inv(cov)
        return self

    def predict(self, s: Sample) -> float:
        if self.mu is None or self.Sinv is None or s.gen.hidden is None or len(s.gen.hidden) == 0:
            return float(_perplexity(s.gen.logprobs))
        idx = self.token_selector(s.gen)
        x = s.gen.hidden[idx].astype(np.float32)
        d = x - self.mu
        md2 = float(d @ self.Sinv @ d)  # Mahalanobis^2
        return md2


# ---------------------------
# Convenience factory & evaluation
# ---------------------------

def make_detector(name: str, **kwargs) -> BaseDetector:
    name = name.lower()
    if name in ["p(true)", "ptrue", "p_true"]:
        return PTrueDetector(**kwargs)
    if name in ["perplexity", "ppx"]:
        return PerplexityDetector()
    if name in ["se", "semantic-entropy"]:
        return SEDetector(**kwargs)
    if name in ["mars"]:
        return MARSDetector(**kwargs)
    if name in ["mars-se", "marsse"]:
        return MARSSEDetector(**kwargs)
    if name in ["ccs"]:
        return CCSDetector(**kwargs)
    if name in ["saplma"]:
        return SAPLMADetector(**kwargs)
    if name in ["haloscope", "halo"]:
        return HaloScopeDetector(**kwargs)
    raise ValueError(f"Unknown detector: {name}")


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Simple AUROC implementation (no sklearn)."""
    # labels: 1 = positive(hallucinated), 0 = negative(correct)
    order = np.argsort(scores)
    sorted_y = labels[order]
    # compute rank-sum AUROC
    # U = sum of ranks for positives - n_pos*(n_pos+1)/2
    n = len(sorted_y)
    ranks = np.arange(1, n + 1)
    n_pos = int(np.sum(sorted_y))
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum_pos = int(np.sum(ranks[sorted_y == 1]))
    U = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(U / (n_pos * n_neg + 1e-12))


# Example usage (pseudo):
# train, test = [...]
# det = make_detector("saplma").fit(train)
# scores = det.predict_many(test)
# print("AUROC=", auroc(scores, np.array([s.label for s in test], dtype=np.int32)))
