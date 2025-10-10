import numpy as np
from tqdm import tqdm
from src.prompt import build_eval_prompt
from src.model import GenConfig


def evaluate(model, dataset, n_samples=5):
    """
    Semantic Entropy (SE) baseline.
    Measures uncertainty via diversity of generations.

    Args:
        model: model object from get_model()
        dataset: list of QA dicts
        n_samples: number of generations per question

    Returns:
        list of dicts [{"label": int, "score": float}, ...]
    """
    results = []

    for item in tqdm(dataset, desc="Semantic Entropy baseline"):
        q = item.get("question", "")
        a_true = item.get("pos") or item.get("answer1") or ""
        a_false = item.get("neg") or item.get("answer2") or ""

        if not a_true or not a_false:
            continue

        # Construct evaluation prompt
        prompt = build_eval_prompt(q, [a_true, a_false])
        cfg = GenConfig(temperature=0.5, top_p=0.9, max_new_tokens=32)

        # Generate multiple samples
        generations = [model.generate(prompt, cfg).text for _ in range(n_samples)]

        # Compute semantic entropy
        unique_ratio = len(set(generations)) / max(len(generations), 1)
        entropy = -np.log(unique_ratio + 1e-8)

        # Higher entropy → more uncertainty → lower confidence in correctness
        # Assign a score inversely proportional to entropy
        results.append({"label": 1, "score": -entropy})
        results.append({"label": 0, "score": entropy})

    return results
