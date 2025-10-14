import numpy as np
from sklearn.metrics import roc_auc_score
import math
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F


def compute_aurocs_with_labels(scores_dict, labels):
    out = {}
    y = np.asarray(labels, dtype=float)
    for k, v in scores_dict.items():
        try:
            s = np.asarray(v, dtype=float)
            out[k] = round(float(roc_auc_score(y, s)), 6)
        except Exception:
            out[k] = float("nan")
    return out


def _um_squad_norm(s: str) -> str:
    import re, string
    s = (s or "").lower().strip()
    s = re.sub(f"[{re.escape(string.punctuation)}]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@torch.inference_mode()
def _um_mean_nll(text: str, tok, model, device) -> float:
    enc = tok(text, return_tensors="pt", add_special_tokens=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    if not hasattr(out, "logits") or out.logits.size(1) < 2:
        return 0.0
    logits = out.logits[:, :-1, :]
    tgt    = enc["input_ids"][:, 1:]
    logp   = F.log_softmax(logits, dim=-1)
    lp     = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return float((-lp.mean()).item())

def _um_get_local_model(tok=None, model=None):
    if tok is not None and model is not None:
        return tok, model
    from src.model import get_model
    import os
    model_dir = os.getenv("MULTI_MODEL_DIR")
    if not model_dir:
        raise RuntimeError("MULTI_MODEL_DIR is not set")
    cli = get_model(model_dir, backend="hf")
    return getattr(cli, "tok"), getattr(cli, "model")

def _um_group_by_q(flat_rows: List[Dict[str, Any]]) -> Dict[int, List[Tuple[int, str]]]:
    bag = defaultdict(list)
    for i, r in enumerate(flat_rows):
        qid = int(r.get("qid") or i)
        a   = r.get("answer") or ""
        bag[qid].append((i, a))
    return bag

def score_p_t(flat_rows: List[Dict[str, Any]], tok=None, model=None) -> List[float]:
    tok, model = _um_get_local_model(tok, model)
    device = next(model.parameters()).device
    out = []
    for r in flat_rows:
        a = r.get("answer") or ""
        mnll = _um_mean_nll(a, tok, model, device)
        out.append(math.exp(-mnll))
    return out

def score_p_s(flat_rows: List[Dict[str, Any]], tok=None, model=None) -> List[float]:
    tok, model = _um_get_local_model(tok, model)
    device = next(model.parameters()).device
    out = []
    for r in flat_rows:
        a = r.get("answer") or ""
        mnll = _um_mean_nll(a, tok, model, device)
        out.append(1.0 / (1.0 + mnll))
    return out

def score_p_c(flat_rows: List[Dict[str, Any]]) -> List[float]:
    by_q = _um_group_by_q(flat_rows)
    out = [0.0] * len(flat_rows)
    for _, rows in by_q.items():
        buckets = defaultdict(list)
        for idx, a in rows:
            buckets[_um_squad_norm(a)].append(idx)
        M = sum(len(v) for v in buckets.values()) or 1
        largest = max((len(v) for v in buckets.values()), default=1)
        pc = largest / float(M)
        for idx, _ in rows:
            out[idx] = pc
    return out