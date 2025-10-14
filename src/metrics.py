import os, json, math
from collections import defaultdict
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def compute_aurocs_with_labels(scores_dict: Dict[str, List[float]], labels: List[int]) -> Dict[str, float]:
    y = np.asarray(labels, dtype=float)
    out = {}
    for k, v in scores_dict.items():
        try:
            s = np.asarray(v, dtype=float)
            out[k] = round(float(roc_auc_score(y, s)), 6)
        except Exception:
            out[k] = float("nan")
    return out


def _squad_norm(s: str) -> str:
    import re, string
    s = (s or "").lower().strip()
    s = re.sub(f"[{re.escape(string.punctuation)}]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _get_local_model(tok=None, model=None):
    if tok is not None and model is not None:
        return tok, model
    from src.model import get_model
    model_dir = os.getenv("MULTI_MODEL_DIR")
    if not model_dir:
        raise RuntimeError("MULTI_MODEL_DIR is not set (local HF model dir required).")
    cli = get_model(model_dir, backend="hf")
    return getattr(cli, "tok"), getattr(cli, "model")


@torch.inference_mode()
def _mean_nll(text: str, tok, model, device) -> float:
    t = (text or "").strip()
    if not t:
        t = "?"
    enc = tok(t + " .", return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask", None)
    if attn is not None:
        attn = attn.to(device)
    out = model(input_ids=ids, attention_mask=attn)
    logits = out.logits[:, :-1, :]
    tgt = ids[:, 1:]
    logp = F.log_softmax(logits, dim=-1)
    lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return float((-lp.mean()).item())


def score_p_t(flat_rows: List[Dict[str, Any]], tok=None, model=None) -> List[float]:
    tok, model = _get_local_model(tok, model)
    device = next(model.parameters()).device
    out = []
    for r in flat_rows:
        m = _mean_nll(r.get("answer") or "", tok, model, device)
        out.append(math.exp(-m))
    return out


def score_p_s(flat_rows: List[Dict[str, Any]], tok=None, model=None) -> List[float]:
    tok, model = _get_local_model(tok, model)
    device = next(model.parameters()).device
    out = []
    for r in flat_rows:
        m = _mean_nll(r.get("answer") or "", tok, model, device)
        out.append(1.0 / (1.0 + math.exp(m)))
    return out


def score_p_c(flat_rows: List[Dict[str, Any]], multi_path: str = "data/squad_multi.json") -> List[float]:
    q2pc = {}
    if os.path.exists(multi_path):
        multi = json.load(open(multi_path))
        for item in multi:
            qid = int(item.get("qid") or 0)
            gens = [g for g in item.get("generations", []) if isinstance(g, str) and g.strip()]
            if len(gens) < 2:
                continue
            buckets = defaultdict(int)
            for g in gens:
                buckets[_squad_norm(g)] += 1
            M = sum(buckets.values())
            if M > 0:
                q2pc[qid] = max(buckets.values()) / float(M)
    out = []
    for r in flat_rows:
        qid = int(r.get("qid") or 0)
        out.append(float(q2pc.get(qid, 0.5)))
    return out
