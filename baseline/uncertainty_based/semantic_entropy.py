# baseline/uncertainty_based/semantic_entropy.py
"""
Semantic Entropy (SE) baseline.

Idea (SE [16]):
  - For each question, sample K answers from the same LLM prompt.
  - Measure pairwise semantic equivalence via mutual entailment (A⇒B and B⇒A).
  - Use GPT (e.g., GPT-3.5) for entailment if available; otherwise fall back
    to local sentence embedding cosine similarity (MiniLM thresholding).
  - Agreement = (# equivalent pairs) / (K * (K - 1) / 2)
  - SE uncertainty = 1 - Agreement
  - The first sampled answer is used as "pred" for EM labeling.

API:
  run(samples, tokenizer, model, build_prompt, cfg, limit=None)
    -> (auroc: float, outputs: List[dict])

Config fields optionally read from `cfg` (with defaults):
  - se_k: int = 5                       # number of samples per question
  - se_use_gpt: bool = True             # prefer GPT for entailment if available
  - se_gpt_model: str = "gpt-3.5-turbo" # GPT model name
  - se_sim_threshold: float = 0.80      # local fallback cosine threshold
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import os
import torch
import torch.nn.functional as F

# absolute import: ensure you launch from project root, e.g., `python -m src.generate`
from baseline.utils import exact_match, auroc_from_pairs


# ----------------------------
# Sampling multiple answers
# ----------------------------
@torch.no_grad()
def _sample_answers(
    tokenizer,
    model,
    prompt: str,
    k: int,
    max_new_tokens: int,
    temperature: float,
) -> List[str]:
    """Sample K answers independently from the same prompt."""
    dev = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(dev)
    in_len = inputs["input_ids"].shape[1]

    outs: List[str] = []
    for _ in range(k):
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )
        text = tokenizer.decode(out_ids[0][in_len:], skip_special_tokens=True)
        outs.append(text.strip())
    return outs


# ----------------------------
# Entailment backends
# ----------------------------
def _have_openai() -> bool:
    try:
        import openai  # noqa: F401
    except Exception:
        return False
    return bool(os.environ.get("OPENAI_API_KEY"))

def _entails_gpt(a: str, b: str, model_name: str = "gpt-3.5-turbo") -> Optional[bool]:
    """
    Use GPT chat completion to decide whether A entails B.
    Return True/False, or None on failure.
    """
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI()  # reads OPENAI_API_KEY / (Azure) base url & version from env
        sys_prompt = (
            "You are an entailment judge. Decide whether statement A logically "
            "entails statement B. Answer strictly with one token: ENTAILS or NOT."
        )
        user_prompt = f"A: {a}\nB: {b}\nAnswer with ENTAILS or NOT."
        resp = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            max_tokens=2,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = (resp.choices[0].message.content or "").strip().upper()
        if text.startswith("ENTAILS"):
            return True
        if text.startswith("NOT"):
            return False
        # fallback parse if model returns slightly different tokens
        if "ENTAIL" in text:
            return True
        if "NOT" in text or "NO" in text:
            return False
        return None
    except Exception:
        return None


_MINILM_TOK = None
_MINILM_MDL = None

def _load_minilm(name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Lazy-load MiniLM for local similarity fallback."""
    global _MINILM_TOK, _MINILM_MDL
    if _MINILM_TOK is not None and _MINILM_MDL is not None:
        return _MINILM_TOK, _MINILM_MDL
    from transformers import AutoTokenizer, AutoModel  # lazy import
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _MINILM_TOK = AutoTokenizer.from_pretrained(name)
    _MINILM_MDL = AutoModel.from_pretrained(name).to(dev).eval()
    return _MINILM_TOK, _MINILM_MDL

@torch.no_grad()
def _embed_sentence(text: str) -> torch.Tensor:
    tok, mdl = _load_minilm()
    dev = next(mdl.parameters()).device
    enc = tok(text, return_tensors="pt", truncation=True, max_length=256).to(dev)
    out = mdl(**enc)
    last = out.last_hidden_state           # [1, L, H]
    mask = enc["attention_mask"].unsqueeze(-1)
    vec = (last * mask).sum(1) / mask.sum(1).clamp(min=1)  # mean pool
    return F.normalize(vec, dim=-1).squeeze(0).detach().cpu()  # [H]

def _entails_local(a: str, b: str, threshold: float = 0.80) -> bool:
    """Symmetric cosine similarity proxy for entailment."""
    va = _embed_sentence(a)
    vb = _embed_sentence(b)
    sim = float(torch.dot(va, vb).item())
    return sim >= threshold


def _mutual_entailment(
    a: str,
    b: str,
    use_gpt: bool,
    gpt_model: str,
    sim_threshold: float,
) -> bool:
    """
    Decide equivalence via mutual entailment: A⇒B and B⇒A.
    Prefer GPT if available; otherwise local fallback.
    """
    if use_gpt and _have_openai():
        ea = _entails_gpt(a, b, model_name=gpt_model)
        eb = _entails_gpt(b, a, model_name=gpt_model)
        if ea is not None and eb is not None:
            return bool(ea and eb)
        # partial/total failure -> fallback
    # local symmetric similarity proxy
    return _entails_local(a, b, threshold=sim_threshold) and _entails_local(b, a, threshold=sim_threshold)


def _agreement_across_answers(
    answers: List[str],
    use_gpt: bool,
    gpt_model: str,
    sim_threshold: float,
) -> float:
    """
    Fraction of unordered pairs that are mutually entailing, in [0,1].
    """
    n = len(answers)
    if n <= 1:
        return 1.0
    total = n * (n - 1) // 2
    agree = 0
    for i in range(n):
        ai = answers[i]
        for j in range(i + 1, n):
            aj = answers[j]
            if _mutual_entailment(ai, aj, use_gpt, gpt_model, sim_threshold):
                agree += 1
    return agree / total if total > 0 else 1.0


# ----------------------------
# Public baseline API
# ----------------------------
def run(
    samples: List[Dict],
    tokenizer,
    model,
    build_prompt,   # (question: str, context: str | None, cfg) -> str
    cfg,            # needs: temperature, max_new_tokens; optional: se_k, se_use_gpt, se_gpt_model, se_sim_threshold
    limit: int | None = None,
) -> Tuple[float, List[Dict]]:
    """
    Execute the SE baseline. Returns (auroc, outputs).
    Positive class for AUROC is hallucination (EM == 0).
    """
    # Config defaults (read from cfg if present)
    K = getattr(cfg, "se_k", 5)
    USE_GPT = bool(getattr(cfg, "se_use_gpt", True))
    GPT_MODEL = getattr(cfg, "se_gpt_model", "gpt-3.5-turbo")
    SIM_THR = float(getattr(cfg, "se_sim_threshold", 0.80))

    data = samples[:limit] if (limit and limit > 0) else samples

    y_true: List[int] = []      # 1 = hallucination, 0 = correct
    y_score: List[float] = []   # SE uncertainty = 1 - agreement
    outputs: List[Dict] = []

    for ex in data:
        q = ex["question"]
        ctx = ex.get("context", "")

        prompt = build_prompt(q, ctx, cfg)
        answers = _sample_answers(
            tokenizer, model, prompt,
            k=K, max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature
        )

        agreement = _agreement_across_answers(
            answers, use_gpt=USE_GPT, gpt_model=GPT_MODEL, sim_threshold=SIM_THR
        )
        se_uncertainty = 1.0 - agreement

        # Use the first sample as the displayed prediction for EM labeling
        pred = answers[0] if answers else ""
        em = exact_match(pred, ex["gold"])  # 1=correct

        y_true.append(1 - em)               # hallucination as positive class
        y_score.append(se_uncertainty)

        outputs.append({
            "question": q,
            "gold": ex["gold"],
            "pred": pred,
            "em_correct": int(em),
            "agreement": float(agreement),
            "uncertainty_se": float(se_uncertainty),
            "answers": answers,  # keep for inspection; remove if size is a concern
        })

    auroc = auroc_from_pairs(y_true, y_score)
    return auroc, outputs
