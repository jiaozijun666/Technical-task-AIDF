import numpy as np
from tqdm import tqdm
from src.prompt import build_eval_prompt
from src.model import GenConfig

def evaluate(model, dataset):
    """
    Baseline: Semantic Entropy (SE)
    Computes semantic diversity among multiple generations.
    """
    results = []
    for item in tqdm(dataset, desc="Semantic Entropy"):
        q = item.get("question", "")
        pos = item.get("gold") or item.get("pos")
        neg = item.get("neg") or item.get("answer2")

        if not pos or not neg:
            continue

        prompt_true = build_eval_prompt(q, [pos])
        prompt_false = build_eval_prompt(q, [neg])
        cfg = GenConfig(temperature=0.5, top_p=0.9, max_new_tokens=32)

        try:
            outs_true = [model.generate(prompt_true, cfg).text for _ in range(3)]
            outs_false = [model.generate(prompt_false, cfg).text for _ in range(3)]
        except Exception as e:
            print(f"[Semantic Entropy] Error: {e}")
            continue

        # estimate entropy
        entropy_true = -np.log(len(set(outs_true)) / 3 + 1e-8)
        entropy_false = -np.log(len(set(outs_false)) / 3 + 1e-8)

        results.append({"label": 1, "score": -entropy_true})
        results.append({"label": 0, "score": -entropy_false})

    return results
