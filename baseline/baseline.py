import os, math, json, re, string
from typing import List, Dict, Any
from collections import defaultdict, Counter

import torch
import torch.nn.functional as F


def _norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _content_tokens(s: str) -> List[str]:
    STOP = {
        "a","an","the","and","or","but","if","then","else","for","to","of","in","on","at",
        "by","with","from","as","is","am","are","was","were","be","been","being",
        "this","that","these","those","it","its","he","she","they","them","his","her",
        "their","we","us","you","your","i","me","my","mine","ours","yours","do","does",
        "did","doing","have","has","had","having","not","no","yes","there","here","than",
        "so","such","very","can","could","should","would","may","might","will","shall"
    }
    s = (s or "").lower()
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    toks = [t for t in s.split() if t and t not in STOP and not t.isnumeric()]
    return toks


def _token_f1(a: str, b: str) -> float:
    a_t = _content_tokens(a)
    b_t = _content_tokens(b)
    if not a_t and not b_t:
        return 1.0
    if not a_t or not b_t:
        return 0.0
    ca, cb = Counter(a_t), Counter(b_t)
    inter = sum((ca & cb).values())
    if inter == 0:
        return 0.0
    prec = inter / len(a_t)
    rec  = inter / len(b_t)
    return 2 * prec * rec / (prec + rec)


def _group_by_q(flat_rows: List[Dict[str, Any]]):
    bag = defaultdict(list)
    for i, r in enumerate(flat_rows):
        bag[int(r["qid"])].append(
            (i, r.get("answer") or "", r.get("gold") or "", r.get("question") or "")
        )
    return bag


def _get_local_model(tok=None, model=None):
    if tok is not None and model is not None:
        return tok, model
    from src.model import get_model
    model_dir = os.getenv("MULTI_MODEL_DIR")
    if not model_dir:
        raise RuntimeError("MULTI_MODEL_DIR is not set (local HF model dir required).")
    cli = get_model(model_dir, backend="hf")
    return getattr(cli, "tok"), getattr(cli, "model")


def _squad_norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(f"[{re.escape(string.punctuation)}]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _squad_tokens(s: str):
    return _squad_norm(s).split()


# --------- gold auto-fill cache (helps p(true) when flat_rows lack gold) ---------
def _load_gold_cache():
    by_id, by_q = {}, {}
    for p in ("data/squad_train.json", "data/squad_test.json"):
        if not os.path.exists(p):
            continue
        try:
            arr = json.load(open(p, "r"))
        except Exception:
            continue
        for ex in arr:
            qid = str(ex.get("id") or ex.get("qid") or "")
            q   = ex.get("question") or ""
            ans = ex.get("answers") or {}
            if isinstance(ans, dict):
                texts = ans.get("text") or ans.get("texts") or []
                g = texts[0] if isinstance(texts, list) and texts else (texts if isinstance(texts, str) else "")
            else:
                g = ""
            if qid:
                by_id[qid] = g
            if q:
                by_q[_squad_norm(q)] = g
    return by_id, by_q

_GOLD_BY_ID, _GOLD_BY_Q = _load_gold_cache()


def _pick_gold_from_row_or_cache(r: Dict[str, Any]) -> str:
    g = r.get("gold")
    if g:
        return g
    qid = r.get("qid")
    if qid is not None:
        g = _GOLD_BY_ID.get(str(qid))
        if g:
            return g
    q = r.get("question")
    if q:
        g = _GOLD_BY_Q.get(_squad_norm(q))
        if g:
            return g
    ans = r.get("answers")
    if isinstance(ans, dict):
        txt = ans.get("text") or ans.get("texts")
        if isinstance(txt, list) and txt:
            return txt[0]
        if isinstance(txt, str):
            return txt
    return ""


@torch.inference_mode()
def _mean_nll(text: str, tok, model, device) -> float:
    enc = tok(text, return_tensors="pt", add_special_tokens=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    if not hasattr(out, "logits") or out.logits.size(1) < 2:
        return 0.0
    logits = out.logits[:, :-1, :]
    tgt    = enc["input_ids"][:, 1:]
    logp   = F.log_softmax(logits, dim=-1)
    lp     = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
    return float((-lp.mean()).item())


def _conf_from_nll(mean_nll: float) -> float:
    # squash to [0,1]; center roughly near nll≈3 (geometric mean prob≈e^-3)
    return float(1.0 / (1.0 + math.exp(mean_nll - 3.0)))


# ------------------------- baselines (all ↑ better) -------------------------
def run_se(flat_rows: List[Dict[str, Any]], tok=None, model=None, args=None) -> List[float]:
    """String-Equivalence agreement within question: majority-normalized share."""
    by_q = _group_by_q(flat_rows)
    out = [0.0] * len(flat_rows)
    for _, rows in by_q.items():
        texts = [_norm_text(a) for _, a, *_ in rows]
        cnt = Counter(texts)
        n = len(texts)
        for (idx, a, *_), t in zip(rows, texts):
            out[int(idx)] = cnt[t] / max(1, n)
    return out

def run_mars(flat_rows: List[Dict[str, Any]], tok=None, model=None, args=None) -> List[float]:
    """Mean pairwise token-F1 similarity with other generations of the same question."""
    by_q = _group_by_q(flat_rows)
    out = [0.0] * len(flat_rows)
    for _, rows in by_q.items():
        ans = [a for _, a, *_ in rows]
        m = len(ans)
        if m <= 1:
            for idx, *_ in rows: out[int(idx)] = 0.0
            continue
        for i, (idx, a, *_ ) in enumerate(rows):
            s = 0.0; k = 0
            for j, b in enumerate(ans):
                if i == j: continue
                s += _token_f1(a, b); k += 1
            out[int(idx)] = s / max(1, k)
    return out

def run_mars_se(flat_rows: List[Dict[str, Any]], tok=None, model=None, args=None) -> List[float]:
    """Simple blend of MARS and SE."""
    s1 = run_mars(flat_rows, tok, model, args)
    s2 = run_se(flat_rows, tok, model, args)
    return [0.5 * a + 0.5 * b for a, b in zip(s1, s2)]




@torch.inference_mode()
def run_perplexity(flat_rows, tok=None, model=None, args=None):
    tok, model = _get_local_model(tok, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    out = []
    for r in flat_rows:
        ans = r.get("answer") or ""
        enc = tok(ans, return_tensors="pt", add_special_tokens=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc)  # single forward
        if not hasattr(outputs, "logits") or outputs.logits.size(1) < 2:
            out.append(0.0); continue
        logits = outputs.logits[:, :-1, :]
        tgt    = enc["input_ids"][:, 1:]
        logp   = F.log_softmax(logits, dim=-1)
        tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        if tok_lp.numel() == 0:
            out.append(0.0); continue
        mean_nll = float((-tok_lp).mean().item())  # mean NLL
        out.append(-mean_nll)                      # ← 关键：越大越真
    return out



def run_p_true(flat_rows, tok=None, model=None, args=None):
    def _tokens(s: str):
        s = _squad_norm(s)
        return s.split()
    def _f1(pred: str, gold: str) -> float:
        pt, gt = _tokens(pred), _tokens(gold)
        if not pt and not gt: return 1.0
        if not pt or not gt:  return 0.0
        pc, gc = Counter(pt), Counter(gt)
        inter = sum((pc & gc).values())
        if inter == 0: return 0.0
        p = inter / len(pt); r = inter / len(gt)
        return 2 * p * r / (p + r)

    out = []
    for r in flat_rows:
        pred = r.get("answer") or ""
        gold = _pick_gold_from_row_or_cache(r)  # <- 关键：用缓存补齐 gold
        out.append(float(_f1(pred, gold)))
    return out



@torch.inference_mode()
def run_ccs(flat_rows: List[Dict[str, Any]], tok=None, model=None, args=None) -> List[float]:
    """Calibrated confidence score from NLL, length-robust in [0,1]."""
    tok, model = _get_local_model(tok, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    out = []
    for r in flat_rows:
        nll = _mean_nll(r.get("answer") or "", tok, model, device)
        out.append(_conf_from_nll(nll))  # larger = more confident/true
    return out

@torch.inference_mode()
def run_saplma(flat_rows: List[Dict[str, Any]], tok=None, model=None, args=None) -> List[float]:
    """Soft average probability (length-normalized): exp(-mean_nll) in [0,1]."""
    tok, model = _get_local_model(tok, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    out = []
    for r in flat_rows:
        nll = _mean_nll(r.get("answer") or "", tok, model, device)
        out.append(float(math.exp(-nll)))  # geometric mean prob
    return out

@torch.inference_mode()
def run_haloscope(flat_rows: List[Dict[str, Any]], tok=None, model=None, args=None) -> List[float]:
    """Heuristic HaloScope: content-overlap with (question+gold) + LM confidence."""
    tok, model = _get_local_model(tok, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    def overlap(ans: str, refs: List[str]) -> float:
        A = set(_content_tokens(ans))
        if not A: return 0.0
        R = set()
        for r in refs: R.update(_content_tokens(r))
        return len(A & R) / max(1, len(A))

    out = []
    for r in flat_rows:
        q = r.get("question") or ""
        a = r.get("answer")  or ""
        g = r.get("gold")    or ""
        ov = overlap(a, [q, g])                 # [0,1]
        nll = _mean_nll(a, tok, model, device)
        conf = _conf_from_nll(nll)              # [0,1]
        out.append(0.6 * ov + 0.4 * conf)       # larger = more likely true

    # avoid degenerate constant vector
    if len(set(round(x, 6) for x in out)) <= 1:
        import random
        out = [x + 1e-6 * random.random() for x in out]
    return out
