# baseline/uncertainty_based/p_true.py
"""
p(True) baseline (uncertainty-based).

Idea:
  1) Generate an answer for each question using the provided prompt builder.
  2) Ask the model to self-report the probability that the answer is factually correct
     (a single number in [0,1]).
  3) Use (1 - p_true) as the hallucination score.
  4) Compute AUROC with hallucination as the positive class (EM == 0).

Inputs:
  - samples: List[dict] with at least {"question": str, "gold": List[str]} and
             optionally {"context": str}
  - tokenizer, model: Hugging Face tokenizer/model (model already on device)
  - build_prompt: callable(question, context, cfg) -> str
  - cfg: object with fields {temperature: float, max_new_tokens: int}
  - limit: optional int to evaluate a subset

Returns:
  (auroc: float, outputs: List[dict])
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import re
import torch

from baseline.utils import generate_and_get_scores, exact_match, auroc_from_pairs


# Deterministic calibration prompt to elicit a single number in [0,1]
CONFIDENCE_PROMPT = (
    "You will be given a question and a candidate answer.\n"
    "Return ONLY a single number between 0 and 1 (inclusive) representing the "
    "probability that the answer is factually correct. Do not include any other text.\n\n"
    "Question: {q}\n"
    "Answer: {a}\n"
    "p(true)="
)

# Max new tokens for the confidence query (small on purpose)
CONF_MAX_NEW_TOKENS = 8


def _parse_probability(text: str) -> float:
    """
    Parse a probability from the model's textual output.
    Accepts formats like:
      - '0.83', '1', '0', '0.0', '1.0'
      - '83%'  (converted to 0.83)
    Falls back to 0.5 if nothing parsable is found.
    """
    s = text.strip()

    # Percentage like '83%'
    m_pct = re.search(r"(\d{1,3})\s*%", s)
    if m_pct:
        val = float(m_pct.group(1)) / 100.0
        return max(0.0, min(1.0, val))

    # Plain float or integer (first match)
    m_num = re.search(r"\b(\d+(?:\.\d+)?)\b", s)
    if m_num:
        val = float(m_num.group(1))
        # If value looks >1, conservatively interpret as a percent
        if val > 1.0:
            val = val / 100.0
        return max(0.0, min(1.0, val))

    return 0.5  # neutral fallback


@torch.no_grad()
def _ask_p_true(tokenizer, model, question: str, answer: str) -> float:
    """
    Ask the model for p(true) with a deterministic decode.
    Returns a float in [0,1] (clamped), or 0.5 on failure.
    """
    try:
        dev = next(model.parameters()).device
        prompt = CONFIDENCE_PROMPT.format(q=question, a=answer.strip())
        inputs = tokenizer(prompt, return_tensors="pt").to(dev)

        out_ids = model.generate(
            **inputs,
            max_new_tokens=CONF_MAX_NEW_TOKENS,
            do_sample=False,   # deterministic
            temperature=0.0,
        )
        new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        p = _parse_probability(raw)
        return max(0.0, min(1.0, p))
    except Exception:
        # Be robust: never crash the pipeline due to parsing/generation issues
        return 0.5


def run(
    samples: List[Dict],
    tokenizer,
    model,
    build_prompt,   # (question, context, cfg) -> str
    cfg,            # PromptConfig-like object
    limit: int | None = None,
) -> Tuple[float, List[Dict]]:
    """
    Execute the p(True) baseline on the provided samples.
    """
    data = samples[:limit] if (limit and limit > 0) else samples

    y_true: List[int] = []     # 1 = hallucination, 0 = correct
    y_score: List[float] = []  # hallucination score = 1 - p_true
    outputs: List[Dict] = []

    for ex in data:
        q = ex["question"]
        ctx = ex.get("context", "")

        # Step 1: generate a candidate answer
        gen_prompt = build_prompt(q, ctx, cfg)
        pred, _perpl, _scores = generate_and_get_scores(
            tokenizer, model, gen_prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )

        # Step 2: ask for p(true)
        p_true = _ask_p_true(tokenizer, model, q, pred)
        h_score = 1.0 - p_true

        # Label via Exact Match against gold answers
        em = exact_match(pred, ex["gold"])  # 1 = correct
        y_true.append(1 - em)               # hallucination as positive class
        y_score.append(h_score)

        outputs.append({
            "question": q,
            "gold": ex["gold"],
            "pred": pred.strip(),
            "em_correct": int(em),
            "p_true": float(p_true),
            "halluc_score": float(h_score),
        })

    auroc = auroc_from_pairs(y_true, y_score)
    return auroc, outputs
