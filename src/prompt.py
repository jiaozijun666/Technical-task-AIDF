def build_eval_prompt(question: str, answers: list[str]) -> str:
    """
    Build an evaluation prompt for LLM-based scoring baselines.
    Used in HaMI, HaMI*, and Semantic Entropy baselines.

    Args:
        question: str — input question text
        answers: list[str] — list of candidate answers (e.g., [A1, A2])

    Returns:
        A formatted string prompt to feed into the model.
    """
    if not answers:
        raise ValueError("answers list cannot be empty")

    parts = [f"Answer {i+1}: {ans.strip()}" for i, ans in enumerate(answers)]
    answers_block = "\n\n".join(parts)

    prompt = (
        f"Question: {question.strip()}\n\n"
        f"{answers_block}\n\n"
        "Which answer is more correct or informative? "
        "Please reply with the index (1, 2, etc.) or the answer text."
    )

    return prompt


def build_generation_prompt(question: str) -> str:
    """
    Build a prompt for open-ended generation tasks (used in multi-sample.py).
    """
    return (
        f"Answer the following question concisely and factually.\n\n"
        f"Question: {question.strip()}\n\n"
        f"Answer:"
    )


def build_hami_prompt(question: str, pos: str, neg: str) -> str:
    """
    Build the comparison prompt used in HaMI baseline.
    """
    return (
        f"Question: {question.strip()}\n\n"
        f"Answer A: {pos.strip()}\n"
        f"Answer B: {neg.strip()}\n\n"
        "Which answer is better? Reply with 'A' or 'B'."
    )


def build_hami_star_prompt(question: str, pos: str, neg: str) -> str:
    """
    Build the comparison prompt used in Enhanced HaMI (HaMI*).
    """
    return (
        f"[Enhanced Evaluation]\n"
        f"Assess which of the following answers is more accurate and informative.\n\n"
        f"Question: {question.strip()}\n\n"
        f"Answer 1: {pos.strip()}\n"
        f"Answer 2: {neg.strip()}\n\n"
        "Your choice: (1/2)"
    )

def get_fact_check_prompt(question: str, answers: list) -> str:
    """
    Build a prompt asking the model to fact-check multiple candidate answers
    and decide which is most factually correct.
    """
    if not answers:
        raise ValueError("answers list cannot be empty")

    answers_block = "\n".join([f"Answer {i+1}: {a}" for i, a in enumerate(answers)])

    prompt = (
        f"Question: {question}\n\n"
        f"{answers_block}\n\n"
        "Which answer is most factually accurate? "
        "Please reply with the answer number only."
    )

    return prompt

__all__ = [
    "build_eval_prompt",
    "build_generation_prompt",
    "build_hami_prompt",
    "build_hami_star_prompt",
    "get_fact_check_prompt"]
