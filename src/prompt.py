# src/prompt.py
"""
Prompt templates for all baselines and HaMI model.
Each function returns a formatted string prompt.
"""

def get_generation_prompt(question: str) -> str:
    """Prompt used in multi_sample.py"""
    return f"Answer the question: {question}"

def get_fact_check_prompt(question: str, gold: str, generation: str) -> str:
    """Prompt used in refined_set.py (GPT-4 factuality judgment)"""
    return f"""
You are a factual accuracy evaluator.

Question: {question}
Reference Answer: {gold}
Model Answer: {generation}

Is the model answer factually correct? 
Reply strictly with:
1 — if correct,
0 — if hallucinated or incorrect.
"""

def get_true_false_prompt(question: str, generation: str) -> str:
    """Prompt used in p_true.py baseline"""
    return f"""
Question: {question}
Answer: {generation}
Is this answer factually correct? True or False?
"""

def get_yes_no_prompt(question: str, gold: str, generation: str) -> str:
    """Prompt used in SAPLMA baseline"""
    return f"""
You are an expert fact checker.

Question: {question}
Candidate Answer: {generation}
Reference Answer: {gold}

Is the candidate answer factually correct? Reply strictly with "Yes" or "No".
"""

def get_hami_instruction(question: str, generation: str, label: int) -> str:
    """Prompt template for HaMI model training"""
    correctness = "factual" if label == 1 else "hallucinated"
    return f"""
Instruction: Classify whether the following answer is factual or hallucinated.

Question: {question}
Answer: {generation}

The answer is {correctness}.
"""
