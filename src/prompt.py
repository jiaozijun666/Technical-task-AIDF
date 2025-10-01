"""
Prompt helpers aligned with the paper (Appendix A.2).
- Default: Zero-shot QA without context (Figure 6 in the paper).
- Optional: QA with context (for debugging/baseline experiments).
- Label judging template (used in the paper with GPT-4.1 to evaluate correctness).
"""

from dataclasses import dataclass


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""
    temperature: float = 0.5
    max_new_tokens: int = 64
    stop: tuple[str, ...] = ()  # optional stop sequences
    add_trailing_space: bool = True  # whether to add a space after "Answer:"


# === Zero-shot QA prompt (paper version, no context) ===
QA_NO_CONTEXT_TEMPLATE = (
    "Answer the following question in a single but complete sentence.\n\n"
    "Question: {question}\n"
    "Answer:{space}"
)

def build_prompt_qa_no_context(question: str, cfg: PromptConfig | None = None) -> str:
    """Build a zero-shot QA prompt without context (used in the paper)."""
    cfg = cfg or PromptConfig()
    space = " " if cfg.add_trailing_space else ""
    return QA_NO_CONTEXT_TEMPLATE.format(question=question.strip(), space=space)


# === QA with context (optional, for baseline/debug) ===
QA_WITH_CONTEXT_TEMPLATE = (
    "Read the context and answer the following question in a single but complete sentence.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:{space}"
)

def build_prompt_qa_with_context(question: str, context: str, cfg: PromptConfig | None = None) -> str:
    """Build a QA prompt that includes context (not used in the paper’s main experiments)."""
    cfg = cfg or PromptConfig()
    space = " " if cfg.add_trailing_space else ""
    return QA_WITH_CONTEXT_TEMPLATE.format(
        question=question.strip(),
        context=context.strip(),
        space=space,
    )


# === Label judging (paper used GPT-4.1 for this) ===
LABEL_JUDGE_TEMPLATE = (
    "You are an evaluator. Determine if the candidate answer is factually correct "
    "with respect to the gold answer. Return only 'CORRECT' or 'INCORRECT'.\n\n"
    "Question: {question}\n"
    "Gold Answer: {gold}\n"
    "Candidate Answer: {pred}\n"
    "Your verdict:"
)

def build_label_judge_prompt(question: str, gold: str, pred: str) -> str:
    """Build a prompt for evaluating correctness of model predictions."""
    return LABEL_JUDGE_TEMPLATE.format(
        question=question.strip(),
        gold=gold.strip(),
        pred=pred.strip(),
    )
