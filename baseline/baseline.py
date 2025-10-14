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



def _load_gold_cache():
    by_id, by_q = {}, {}
    sources = [
        "data/squad_train.json",
        "data/squad_test.json",
    ]
    for p in sources:
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
            g = ""
            if isinstance(ans, dict):
                texts = ans.get("text") or ans.get("texts") or []
                g = texts[0] if isinstance(texts, list) and texts else (texts if isinstance(texts, str) else "")
            if qid and g:
                by_id[qid] = g
            if q and g:
                by_q[_squad_norm(q)] = g
    return by_id, by_q


_GOLD_BY_ID, _GOLD_BY_Q = _load_gold_cache()



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
def run_p_true(flat_rows, tok=None, model=None, args=None):
    import os, time, re, math, random
    from collections import defaultdict
    from src.api import chat_complete
    mdl = os.getenv("PTRUE_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1"))
    max_tokens = int(os.getenv("PTRUE_MAX_TOKENS", "16"))
    top_p = float(os.getenv("PTRUE_TOP_P", "0.95"))
    temperature = 0.5
    retries = int(os.getenv("PTRUE_RETRIES", "2"))
    sleep_sec = float(os.getenv("PTRUE_SLEEP", "0.0"))
    log_every = int(os.getenv("PTRUE_LOG_INTERVAL", os.getenv("MULTI_LOG_INTERVAL", "50")))
    only_top1 = os.getenv("PTRUE_ONLY_TOP1", "0") == "1"
    sample_k = int(os.getenv("PTRUE_SAMPLE_K", "0"))
    tok, model = _get_local_model(tok, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    def _prompt(q, c, a):
        return (
            "You are a precise factuality judge.\n"
            "Given QUESTION, optional CONTEXT, and ANSWER, output a single probability in [0,1] that the answer is factually correct with respect to the question (and context if provided).\n"
            "Output strictly as: score: <number between 0 and 1>\n\n"
            f"QUESTION: {q}\nCONTEXT: {c}\nANSWER: {a}\nscore:"
        )

    def _remote(q, c, a):
        backoff = 1.0
        for _ in range(max(1, retries + 1)):
            try:
                txt = chat_complete(prompt=_prompt(q, c, a), model=mdl, max_tokens=max_tokens, top_p=top_p, temperature=temperature)
                m = re.search(r"([01](?:\.\d+)?)", txt)
                if m:
                    v = float(m.group(1))
                    if v < 0.0: v = 0.0
                    if v > 1.0: v = 1.0
                    return v
            except Exception:
                pass
            time.sleep(backoff)
            backoff *= 2.0
        return None

    def _fallback(a):
        nll = _mean_nll(a or "", tok, model, device)
        z = -nll
        return float(1.0 / (1.0 + math.exp(-z)))

    by_q = defaultdict(list)
    for i, r in enumerate(flat_rows):
        by_q[int(r["qid"])].append(i)

    select = set()
    if only_top1:
        for _, idxs in by_q.items():
            select.add(min(idxs))
    elif sample_k > 0:
        rng = random.Random(0)
        for _, idxs in by_q.items():
            take = idxs if len(idxs) <= sample_k else rng.sample(idxs, sample_k)
            select.update(take)
    else:
        for idxs in by_q.values():
            select.update(idxs)

    out = [0.0] * len(flat_rows)
    total = len(select)
    done = 0
    for i, r in enumerate(flat_rows):
        q = r.get("question") or ""
        c = r.get("context") or r.get("ctx") or ""
        a = r.get("answer") or ""
        if i in select:
            v = _remote(q, c, a)
            if v is None:
                v = _fallback(a)
            out[i] = float(v)
            done += 1
            if log_every and (done % log_every == 0):
                print(f"[p_true] {done}/{total}")
            if sleep_sec > 0:
                time.sleep(sleep_sec)
        else:
            out[i] = float(_fallback(a))
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

@torch.inference_mode()
def run_perplexity(flat_rows, tok=None, model=None, args=None):
    tok, model = _get_local_model(tok, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    out = []
    for r in flat_rows:
        nll = _mean_nll(r.get("answer") or "", tok, model, device)
        out.append(float(-nll))
    return out


@torch.inference_mode()
def run_ccs(flat_rows, tok=None, model=None, args=None):
    tok, model = _get_local_model(tok, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    nlls = []
    for r in flat_rows:
        nlls.append(_mean_nll(r.get("answer") or "", tok, model, device))
    from collections import defaultdict
    by_q = defaultdict(list)
    for i, r in enumerate(flat_rows):
        by_q[int(r["qid"])].append(i)
    out = [0.0] * len(flat_rows)
    for idxs in by_q.values():
        vals = [nlls[i] for i in idxs]
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1)
        sigma = var ** 0.5
        for i in idxs:
            z = 0.0 if sigma == 0 else (nlls[i] - mu) / sigma
            out[i] = float(-z)
    return out


@torch.inference_mode()
def run_saplma(flat_rows, tok=None, model=None, args=None):
    tok, model = _get_local_model(tok, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    import string
    STOP = {
        "a","an","the","and","or","but","if","then","else","for","to","of","in","on","at","by","with",
        "from","as","is","am","are","was","were","be","been","being","this","that","these","those",
        "it","its","he","she","they","them","his","her","their","we","us","you","your","i","me","my",
        "do","does","did","have","has","had","not","no","yes","there","here","than","so","such","very"
    }
    def _content_mask(enc):
        ids = enc["input_ids"][0].tolist()
        toks = tok.convert_ids_to_tokens(ids)
        keep = []
        for t in toks:
            s = (t or "").lower().strip()
            s = "".join(ch for ch in s if ch not in set(string.punctuation))
            keep.append(bool(s) and s not in STOP and not s.isnumeric())
        if len(keep) < len(ids): keep += [False] * (len(ids) - len(keep))
        if len(keep) > 1: keep = keep[1:]
        else: keep = []
        return torch.tensor(keep, device=device, dtype=torch.bool)

    out = []
    for r in flat_rows:
        a = r.get("answer") or ""
        q = r.get("question") or ""

        enc_a = tok(a, return_tensors="pt", add_special_tokens=True).to(device)
        out_a = model(**enc_a)
        logp_a = F.log_softmax(out_a.logits[:, :-1, :], dim=-1)
        tgt_a  = enc_a["input_ids"][:, 1:]
        lp_seq = logp_a.gather(-1, tgt_a.unsqueeze(-1)).squeeze(-1)
        m = _content_mask(enc_a)
        lp_a = lp_seq.mean() if m.numel() == 0 or not m.any() else lp_seq[0][m].mean()

        if q:
            enc_q = tok(q, return_tensors="pt", add_special_tokens=True).to(device)
            out_q = model(**enc_q)
            logp_q = F.log_softmax(out_q.logits[:, :-1, :], dim=-1)
            tgt_q  = enc_q["input_ids"][:, 1:]
            lp_q   = logp_q.gather(-1, tgt_q.unsqueeze(-1)).squeeze(-1).mean()
        else:
            lp_q = torch.zeros((), device=device)

        out.append(float((lp_a - lp_q).item()))
    return out
