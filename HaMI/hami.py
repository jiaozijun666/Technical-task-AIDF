import os
from typing import List, Dict, Any
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

STRATEGY = os.getenv("HAMI_STRATEGY", "last")      
TOP_RATIO = float(os.getenv("HAMI_TOP_RATIO", "0.1"))
LAYER = int(os.getenv("HAMI_LAYER", "0"))          


class HaMI(nn.Module):
    """
    Detector head (same as HaMI*):
      Linear(d -> 256) -> BN -> ReLU -> Dropout(0.3) -> Linear(256 -> 1) -> Sigmoid
    Produces a per-token score in [0, 1].
    """
    def __init__(self, n_features: int, hidden: int = 256, p_drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, token_feats: torch.Tensor, layer_idx: int) -> torch.Tensor:
        # token_feats: [T, n_layers, d]
        x = token_feats[:, layer_idx, :].float()
        return self.net(x).flatten()  # [T]

def _norm_text(s: str) -> str:
    import re, string
    s = (s or "").lower().strip()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _majority_key(texts: List[str]) -> str:
    from collections import Counter
    c = Counter(_norm_text(t) for t in texts)
    return c.most_common(1)[0][0] if c else ""


def _sem_consistency_ratio(texts: List[str]) -> float:
    """
    Proxy of semantic-consistency fraction P^c:
      majority cluster proportion by normalized string equality.
    """
    if not texts:
        return 0.0
    from collections import Counter
    cnt = Counter(_norm_text(t) for t in texts)
    return max(cnt.values()) / max(1, len(texts))


@torch.inference_mode()
def _answer_hidden_and_probs(answer: str, tok, model):
    """
    Encode an answer and return:
      feats:  [seq, n_layers, hidden]
      tok_p:  [seq] token probabilities for the next-token (approx.)
      s_nll:  float, mean negative log-likelihood over tokens
    """
    enc = tok(answer, return_tensors="pt", add_special_tokens=True)
    out = model(
        **{k: v.to(model.device) for k, v in enc.items()},
        output_hidden_states=True,
        use_cache=False,
    )

    layers = [h[0] for h in out.hidden_states]             # list of [seq, hidden]
    feats = torch.stack(layers, dim=0).permute(1, 0, 2)    # [seq, n_layers, hidden]
    feats = feats.contiguous().cpu()

    # token-level probabilities (greedy proxy w.r.t given tokens)
    if hasattr(out, "logits"):
        logits = out.logits[0]                             # [seq, vocab]
        probs = F.softmax(logits, dim=-1)                  # [seq, vocab]
        ids   = enc["input_ids"][0].to(logits.device)      # [seq]
        tok_p = probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1).detach().cpu()  # [seq]
        s_nll = float((-torch.log(tok_p.clamp_min(1e-12))).mean().item())
    else:
        tok_p = torch.ones(feats.size(0))
        s_nll = 0.0

    return feats, tok_p, s_nll


def _topk_mean(seq_scores: torch.Tensor, lengths: List[int], r_k: float) -> List[float]:
    """
    For each sequence (length L), compute mean of top-k token scores, k = floor(r_k * L) + 1.
    """
    vals, start = [], 0
    for L in lengths:
        s = seq_scores[start:start + L]
        start += L
        k = int(L * r_k) + 1
        if k >= L:
            vals.append(float(s.mean().item()))
        else:
            vals.append(float(torch.topk(s, k).values.mean().item()))
    return vals


def _group_by_q(flat_rows: List[Dict[str, Any]]):
    """
    Group rows by question id. Each item: (row_idx, answer_text, gold_text).
    """
    by_q = defaultdict(list)
    for idx, r in enumerate(flat_rows):
        by_q[int(r["qid"])].append((idx, r.get("answer") or "", r.get("gold") or ""))
    return by_q

@torch.inference_mode()
def run_hami(
    flat_rows: List[Dict[str, Any]],
    tok=None,
    model=None,
    args=None,
) -> List[float]:
    """
    HaMI scoring with uncertainty scaling (Eq. 8 in the paper):
      - Detector head identical to HaMI*.
      - Token features h are scaled by (1 + λ·P), where P is chosen by HAMI_UNCERT:
          'c' : semantic-consistency fraction  P^c  (default; proxy by normalized majority)
          's' : sentence-level uncertainty      P^s  (mean NLL)
          't' : token-level uncertainty         P^t  (per-token probability)
          'none' : no scaling (equivalent to HaMI*)
      - Aggregate per-generation score with top-k tokens.
    Returns one score per row in `flat_rows` (aligned by index).
    """
    # Load local model if not provided
    if tok is None or model is None:
        from src.model import get_model
        model_dir = os.getenv("MULTI_MODEL_DIR")
        if not model_dir:
            raise RuntimeError("MULTI_MODEL_DIR is not set for HaMI. Point it to your local HF model dir.")
        cli = get_model(model_dir, backend="hf")
        tok, model = getattr(cli, "tok"), getattr(cli, "model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    # Hyper-parameters
    layer_idx   = int(os.getenv("HAMI_LAYER", "-1"))
    r_k         = float(os.getenv("HAMI_RK", "0.1"))        # k = floor(r_k * L) + 1
    uncert_type = os.getenv("HAMI_UNCERT", "c").lower()     # 'c' | 's' | 't' | 'none'
    lam         = float(os.getenv("HAMI_LAMBDA", "1.0"))

    hidden_size = int(model.config.hidden_size)
    detector = HaMI(n_features=hidden_size).to(device).eval()

    out_scores = [0.0] * len(flat_rows)
    by_q = _group_by_q(flat_rows)

    for _, rows in by_q.items():
        texts = [a for _, a, _ in rows]
        maj = _majority_key(texts)

        # ---- split into normal (majority) / abnormal by normalized text
        n_idxs, a_idxs, n_texts, a_texts = [], [], [], []
        for idx, a, _ in rows:
            if _norm_text(a) == maj:
                n_idxs.append(int(idx)); n_texts.append(a)
            else:
                a_idxs.append(int(idx)); a_texts.append(a)

        # ensure non-empty normal set
        if not n_texts and a_texts:
            n_texts, a_texts = a_texts, []
            n_idxs, a_idxs   = a_idxs, []

        if not n_texts:
            for idx, _, _ in rows:
                out_scores[int(idx)] = 0.0
            continue

        # ---- compute uncertainty scalar P for this question (if needed)
        if uncert_type == "c":
            P_scalar = _sem_consistency_ratio(texts)  # in [0,1]
        else:
            P_scalar = 0.0

        # ---- extract features and apply uncertainty scaling
        n_feats, a_feats = [], []
        for a in n_texts:
            feats, tok_p, s_nll = _answer_hidden_and_probs(a, tok, model)
            if uncert_type == "t":
                feats = feats * (1.0 + lam * tok_p.unsqueeze(-1).clamp(0.0, 1.0))
            elif uncert_type == "s":
                feats = feats * (1.0 + lam * max(0.0, s_nll))
            elif uncert_type == "c":
                feats = feats * (1.0 + lam * P_scalar)
            # else 'none': no scaling
            n_feats.append(feats)

        for a in a_texts:
            feats, tok_p, s_nll = _answer_hidden_and_probs(a, tok, model)
            if uncert_type == "t":
                feats = feats * (1.0 + lam * tok_p.unsqueeze(-1).clamp(0.0, 1.0))
            elif uncert_type == "s":
                feats = feats * (1.0 + lam * max(0.0, s_nll))
            elif uncert_type == "c":
                feats = feats * (1.0 + lam * P_scalar)
            a_feats.append(feats)

        lens_n = [f.shape[0] for f in n_feats]
        lens_a = [f.shape[0] for f in a_feats]

        parts = [torch.cat(n_feats, dim=0)]
        if a_feats:
            parts.append(torch.cat(a_feats, dim=0))
        feats_all = torch.cat(parts, dim=0).to(device)      # [sum(L), n_layers, d]

        # ---- token scoring and top-k aggregation
        token_scores = detector(feats_all, layer_idx)       # [sum(L)]
        n_total = sum(lens_n)

        n_seq_scores = _topk_mean(token_scores[:n_total], lens_n, r_k)
        for idx, sc in zip(n_idxs, n_seq_scores):
            out_scores[int(idx)] = float(sc)

        if a_feats:
            a_seq_scores = _topk_mean(token_scores[n_total:], lens_a, r_k)
            for idx, sc in zip(a_idxs, a_seq_scores):
                out_scores[int(idx)] = float(sc)

    return out_scores
