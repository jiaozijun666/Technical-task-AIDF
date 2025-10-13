# src/prompt.py
# Prompts aligned with the paper:
# - Generation: zero-shot QA without context
# - Evaluation (GPT-4.1): label correctness with a strict JSON output
# Labels: 0 = correct/trustworthy, 1 = hallucinated/incorrect

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


# ---------- Generation (no-context, zero-shot) ----------
# Figure 6 style: “QA without context, answer concisely.”
# The temperature=0.5 setting should be enforced in the calling code.
GEN_ZERO_SHOT = (
    "Answer the question concisely. Do not include explanations unless necessary.\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
)

def build_generation_prompt(question: str) -> str:
    """
    Build the zero-shot, context-free QA prompt used to elicit an answer.
    Paper setting: no context, concise answer. Temperature controlled by caller.
    """
    q = (question or "").strip()
    return GEN_ZERO_SHOT.format(question=q)


# ---------- Evaluation (GPT-4.1) ----------
# Figure 7 style: ask GPT-4.1 to judge correctness based on gold answer (if provided)
# and its own knowledge. Output strictly formatted JSON to enable programmatic parsing.

EVAL_PRIMARY_SYS = (
    "You are a strict evaluator of factual correctness for short QA outputs."
)

EVAL_PRIMARY_USER = (
    "Determine whether the model's answer is factually correct.\n"
    "Use the gold answer if provided; otherwise rely on your own knowledge.\n\n"
    "Rules:\n"
    "1) Output strictly in JSON with keys: label, reason.\n"
    "2) label MUST be 0 or 1 only.\n"
    "   - 0 = correct / trustworthy\n"
    "   - 1 = hallucinated / incorrect\n"
    "3) reason MUST be a short, evidence-based justification (one or two sentences).\n"
    "4) Be conservative: if the answer asserts an incorrect fact or contradicts the gold, label=1.\n\n"
    "Question:\n{question}\n\n"
    "Model Answer:\n{answer}\n\n"
    "Gold Answer (may be empty):\n{gold}\n\n"
    "Now produce ONLY the JSON object (no prose)."
)

def build_eval_primary_prompt(question: str, answer: str, gold: Optional[str]) -> tuple[str, str]:
    """
    First-pass evaluation prompt for GPT-4.1.
    Returns (system_prompt, user_prompt).
    """
    q = (question or "").strip()
    a = (answer or "").strip()
    g = (gold or "").strip() if gold is not None else ""
    return EVAL_PRIMARY_SYS, EVAL_PRIMARY_USER.format(question=q, answer=a, gold=g)


# In the paper, items judged positive (hallucinated) are re-checked once; if the second
# judgment disagrees, the item is discarded. This prompt performs that re-judgment.
EVAL_REJUDGE_SYS = (
    "You are re-judging factual correctness for QA outputs. Be careful and consistent."
)

EVAL_REJUDGE_USER = (
    "This item was previously labeled as hallucinated/incorrect (label=1). "
    "Re-evaluate carefully. If you agree it is incorrect, keep label=1; "
    "if you are convinced it is correct, use label=0.\n\n"
    "Rules:\n"
    "1) Output strictly in JSON with keys: label, reason.\n"
    "2) label MUST be 0 or 1 only.\n"
    "   - 0 = correct / trustworthy\n"
    "   - 1 = hallucinated / incorrect\n"
    "3) reason MUST briefly justify your decision.\n\n"
    "Question:\n{question}\n\n"
    "Model Answer:\n{answer}\n\n"
    "Gold Answer (may be empty):\n{gold}\n\n"
    "Now produce ONLY the JSON object."
)

def build_eval_rejudge_prompt(question: str, answer: str, gold: Optional[str]) -> tuple[str, str]:
    """
    Second-pass re-judgment prompt for items initially labeled positive (label=1).
    Returns (system_prompt, user_prompt).
    """
    q = (question or "").strip()
    a = (answer or "").strip()
    g = (gold or "").strip() if gold is not None else ""
    return EVAL_REJUDGE_SYS, EVAL_REJUDGE_USER.format(question=q, answer=a, gold=g)


# ---------- Optional small helper ----------
@dataclass
class EvalInput:
    question: str
    answer: str
    gold: Optional[str] = None

def build_eval_prompts(e: EvalInput, rejudge: bool = False) -> tuple[str, str]:
    """
    Convenience wrapper to get (system, user) for primary or re-judge evaluation.
    """
    if rejudge:
        return build_eval_rejudge_prompt(e.question, e.answer, e.gold)
    return build_eval_primary_prompt(e.question, e.answer, e.gold)
