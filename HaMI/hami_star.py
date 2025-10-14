import os
from typing import List, Dict, Any
from collections import defaultdict
import torch
import torch.nn as nn
class HaMI(nn.Module):
    """
    HaMI* detector head (paper baseline):
      Linear(d -> 256) -> BN -> ReLU -> Dropout(0.3) -> Linear(256 -> 1) -> Sigmoid
    Produces a per-token anomaly score in [0, 1].
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
        return self.net(x).flatten()  

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


@torch.inference_mode()
def _answer_hidden(answer: str, tok, model) -> torch.Tensor:
    """
    Encode an answer and return token-level hidden features of all layers:
      return shape: [seq_len, n_layers, hidden]
    """
    enc = tok(answer, return_tensors="pt", add_special_tokens=True)
    out = model(
        **{k: v.to(model.device) for k, v in enc.items()},
        output_hidden_states=True,
        use_cache=False,
    )
    # hidden_states: tuple(n_layers) of [1, seq, hidden]
    layers = [h[0] for h in out.hidden_states]            # list of [seq, hidden]
    feats = torch.stack(layers, dim=0).permute(1, 0, 2)   # [seq, n_layers, hidden]
    return feats.cpu()


def _topk_mean(seq_scores: torch.Tensor, lengths: List[int], r_k: float) -> List[float]:
    """
    For each sequence (length L), compute mean of top-k token scores, where k = floor(r_k * L) + 1.
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
def run_hami_star(
    flat_rows: List[Dict[str, Any]],
    tok=None,
    model=None,
    args=None,
) -> List[float]:
    """
    HaMI* scoring:
      1) Per question, split generations into a 'normal' cluster (majority by normalized text) vs 'abnormal'.
      2) Extract token features from the specified transformer layer for each generation.
      3) Pass token features through HaMI head to get token-level scores.
      4) Aggregate per-generation score as mean of top-k tokens (k = floor(r_k * L) + 1).
    Returns a score for every row in `flat_rows` (same length, aligned by index).
    """
    # Load local model if not provided
    if tok is None or model is None:
        from src.model import get_model
        model_dir = os.getenv("MULTI_MODEL_DIR")
        if not model_dir:
            raise RuntimeError("MULTI_MODEL_DIR is not set for HaMI*. Point it to your local HF model dir.")
        cli = get_model(model_dir, backend="hf")
        tok, model = getattr(cli, "tok"), getattr(cli, "model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    # Hyperparameters (paper defaults): layer=-1 (last), r_k=0.1
    layer_idx = int(os.getenv("HAMI_LAYER", "-1"))
    r_k       = float(os.getenv("HAMI_RK", "0.1"))

    # Detector head
    hidden_size = int(model.config.hidden_size)
    detector = HaMI(n_features=hidden_size).to(device).eval()

    out_scores = [0.0] * len(flat_rows)
    by_q = _group_by_q(flat_rows)

    for _, rows in by_q.items():
        texts = [a for _, a, _ in rows]
        maj = _majority_key(texts)

        # Split by majority-normalized text (normal vs abnormal)
        n_idxs, a_idxs, n_texts, a_texts = [], [], [], []
        for idx, a, _ in rows:
            if _norm_text(a) == maj:
                n_idxs.append(int(idx))
                n_texts.append(a)
            else:
                a_idxs.append(int(idx))
                a_texts.append(a)

        # Ensure we have a non-empty "normal" set
        if not n_texts and a_texts:
            n_texts, a_texts = a_texts, []
            n_idxs, a_idxs   = a_idxs, []

        # If still empty, zero-score all rows of this question
        if not n_texts:
            for idx, _, _ in rows:
                out_scores[int(idx)] = 0.0
            continue

        # Encode answers to token features
        n_feats = [_answer_hidden(t, tok, model) for t in n_texts]
        a_feats = [_answer_hidden(t, tok, model) for t in a_texts] if a_texts else []

        lens_n = [f.shape[0] for f in n_feats]
        lens_a = [f.shape[0] for f in a_feats]

        # Concatenate token features: normal then abnormal
        parts = [torch.cat(n_feats, dim=0)]
        if a_feats:
            parts.append(torch.cat(a_feats, dim=0))
        feats_all = torch.cat(parts, dim=0).to(device)  # [sum(L), n_layers, d]

        # Token-level scores and top-k aggregation per generation
        token_scores = detector(feats_all, layer_idx)  # [sum(L)]
        n_total = sum(lens_n)
        n_seq_scores = _topk_mean(token_scores[:n_total], lens_n, r_k)
        for idx, sc in zip(n_idxs, n_seq_scores):
            out_scores[int(idx)] = float(sc)

        if a_feats:
            a_seq_scores = _topk_mean(token_scores[n_total:], lens_a, r_k)
            for idx, sc in zip(a_idxs, a_seq_scores):
                out_scores[int(idx)] = float(sc)

    return out_scores

