# baseline/uncertainty_based/perplexity.py
"""
Perplexity baseline (faithful to Eq.(6) in the paper).

Uncertainty score = mean negative log-likelihood (NLL) of the actually generated
tokens y_1..y_T:
    score = -(1/T) * sum_t log P(y_t | y_<t, x)
Higher score => more likely hallucination.
AUROC is computed with hallucination as the positive class (EM == 0).
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F

# use absolute imports to avoid relative-import issues
from baseline.utils import exact_match, auroc_from_pairs


@torch.no_grad()
def _gen_and_mean_nll(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> Tuple[str, float]:
    """
    Generate one answer and compute the mean NLL of the generated token sequence
    exactly as in Eq.(6): -(1/T) * sum_t log P(y_t | y_<t, x)

    Returns:
        pred_text: decoded continuation
        mean_nll:  float (higher => more uncertain)
    """
    dev = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(dev)
    in_len = enc["input_ids"].shape[1]

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        output_scores=True,              # logits for each generated step
        return_dict_in_generate=True,
    )

    # decode generated continuation
    gen_ids = out.sequences[0][in_len:]                 # [T]
    pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # out.scores is a list of length T; each item is logits over vocab for step t
    scores = out.scores                                  # List[Tensor [1, vocab]]
    T = len(scores)
    if T == 0:
        return pred_text, 0.0

    # compute log-prob of the ACTUAL sampled token at each step
    logps = []
    for t, logits in enumerate(scores):
        log_prob = F.log_softmax(logits, dim=-1)[0, gen_ids[t].item()]
        logps.append(log_prob)
    logps = torch.stack(logps)                           # [T]
    mean_nll = float((-logps).mean().item())            # -(1/T) * sum log p

    return pred_text, mean_nll


def run(
    samples: List[Dict],
    tokenizer,
    model,
    build_prompt,   # (question: str, context: str | None, cfg) -> str
    cfg,            # object with fields {temperature: float, max_new_tokens: int}
    limit: int | None = None,
) -> Tuple[float, List[Dict]]:
    """
    Execute the sentence-level perplexity baseline (Eq. 6).
    """
    data = samples[:limit] if (limit and limit > 0) else samples

    y_true: List[int] = []     # 1 = hallucination, 0 = correct
    y_score: List[float] = []  # uncertainty score = mean NLL of generated tokens
    outputs: List[Dict] = []

    for ex in data:
        q = ex["question"]
        ctx = ex.get("context", "")

        prompt = build_prompt(q, ctx, cfg)
        pred, mean_nll = _gen_and_mean_nll(
            tokenizer, model, prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )

        em = exact_match(pred, ex["gold"])  # 1=correct
        y_true.append(1 - em)               # hallucination as positive class
        y_score.append(mean_nll)

        outputs.append({
            "question": q,
            "gold": ex["gold"],
            "pred": pred.strip(),
            "em_correct": int(em),
            "uncertainty_mean_nll": float(mean_nll),
            # NOTE: if you also want classic perplexity: math.exp(mean_nll)
        })

    auroc = auroc_from_pairs(y_true, y_score)
    return auroc, outputs
