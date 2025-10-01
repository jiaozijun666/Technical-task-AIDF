"""
p(True) baseline: ask the LLM to self-report the probability that its answer
is factually correct. We:
  1) Generate an answer to the question (using the provided prompt builder).
  2) Ask a follow-up calibration prompt that MUST output a single number in [0,1].
  3) Parse that number as p_true and use (1 - p_true) as the hallucination score.

Positive class for AUROC = hallucination (EM == 0).
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import re
import torch

from baseline.utils import generate_and_get_scores, exact_match, auroc_from_pairs


# A strict calibration prompt to elicit a single number in [0,1]
CONFIDENCE_PROMPT = (
    "You will be given a question and a candidate answer.\n"
    "Return ONLY a single number between 0 and 1 (inclusive) representing the "
    "probability that the answer is factually correct. Do not include any other text.\n\n"
    "Question: {q}\n"
    "Answer: {a}\n"
    "p(true)="
)


def _parse_probability(text: str) -> float:
    """
    Extract a probability from model text. Accepts:
      - '0.83', '1', '0', '0.0', '1.0'
      - '83%' (converted to 0.83)
    Falls back to 0.5 if nothing parsable is found.
    """
    s = text.strip()

    # percentage like '83%'
    m_pct = re.search(r"(\d{1,3})\s*%", s)
    if m_pct:
        val = float(m_pct.group(1)) / 100.0
        return max(0.0, min(1.0, val))

    # plain float or integer
    m_num = re.search(r"\b(\d+(?:\.\d+)?)\b", s)
    if m_num:
        val = float(m_num.group(1))
        # If it looks like a percent (e.g., 83), map to 0.83 conservatively.
        if val > 1.0:
            val = val / 100.0
        return max(0.0, min(1.0, val))

    return 0.5


@torch.no_grad()
def _ask_p_true(tokenizer, model, question: str, answer: str) -> float:
    """
    Query the model for p(true) with a deterministic decode (no sampling).
    """
    dev = next(model.parameters()).device
    prompt = CONFIDENCE_PROMPT.format(q=question, a=answer.strip())
    inputs = tokenizer(prompt, return_tensors="pt").to(dev)

    out_ids = model.generate(
        **inputs,
        max_new_tokens=8,       # a few tokens are enough for a number
        do_sample=False,        # deterministic for stability
        temperature=0.0,
    )
    new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return _parse_probability(text)


def run(
    samples: List[Dict],
    tokenizer,
    model,
    build_prompt,   # callable: (question, context, cfg) -> str
    cfg,            # PromptConfig
    limit: int | None = None,
) -> Tuple[float, List[Dict]]:
    """
    Execute the p(True) baseline.
    Returns:
      auroc: float
      outputs: per-example logs
    """
    data = samples[:limit] if (limit and limit > 0) else samples

    y_true: List[int] = []     # 1 = hallucination, 0 = correct
    y_score: List[float] = []  # hallucination score = 1 - p_true
    outputs: List[Dict] = []

    for ex in data:
        q = ex["question"]
        ctx = ex.get("context", "")

        # Step 1: generate a candidate answer (also returns a perplexity proxy we ignore here)
        gen_prompt = build_prompt(q, ctx, cfg)
        pred, _, _ = generate_and_get_scores(
            tokenizer, model, gen_prompt, cfg.max_new_tokens, cfg.temperature
        )

        # Step 2: ask for p(true) on the generated answer
        p_true = _ask_p_true(tokenizer, model, q, pred)
        h_score = 1.0 - p_true

        # Label via Exact Match against gold answers
        em = exact_match(pred, ex["gold"])  # 1=correct, 0=hallucination
        y_true.append(1 - em)               # hallucination as positive class
        y_score.append(h_score)

        outputs.append({
            "question": q,
            "gold": ex["gold"],
            "pred": pred.strip(),
            "em_correct": int(em),
            "p_true": p_true,
            "halluc_score": h_score,
        })

    auroc = auroc_from_pairs(y_true, y_score)
    return auroc, outputs
