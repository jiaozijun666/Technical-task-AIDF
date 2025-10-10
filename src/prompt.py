"""
prompt.py — Unified prompt templates for all baselines and HaMI variants
"""

def build_eval_prompt(question, answers=None):
    """
    For general evaluation or generation tasks.
    Used by: multi_sample.py, semantic_entropy.py, etc.
    """
    prompt = f"Question: {question.strip()}\n"
    if answers:
        prompt += "\n".join([f"Answer {i+1}: {a}" for i, a in enumerate(answers)])
        prompt += "\nPlease evaluate which answer is more factually correct."
    else:
        prompt += "Please provide an accurate, factual, and concise answer."
    return prompt.strip()


def build_hami_prompt(question, pos_answer, neg_answer):
    """
    HaMI prompt — used for adaptive token selection with uncertainty weighting.
    The model is expected to form internal factual reasoning representations.
    """
    return f"""You are a knowledgeable fact-verification expert.

Question: {question.strip()}

Candidate Answer A (possibly correct): {pos_answer.strip()}
Candidate Answer B (possibly hallucinated): {neg_answer.strip()}

Reason internally about the factual alignment between the question and each answer.
Return your internal representation (hidden states), not a textual answer.
"""


def build_hami_star_prompt(question, pos_answer, neg_answer):
    """
    HaMI* prompt — ablation version without uncertainty weighting.
    Only focuses on factual consistency.
    """
    return f"""Question: {question.strip()}

Answer A: {pos_answer.strip()}
Answer B: {neg_answer.strip()}

Assess which answer is factually consistent with the question.
Return only the internal representation, not a textual response.
"""


def build_pairwise_prompt(question, pos_answer, neg_answer):
    """
    Simple pairwise comparison prompt.
    Used in baselines such as Hami prompt ablations or contrastive setups.
    """
    return f"Q: {question}\nA1: {pos_answer}\nA2: {neg_answer}\nWhich answer is more factually accurate?"
