import numpy as np
from tqdm import tqdm
from src.utils import calculate_temperature_scaled_agreement

def evaluate(model, dataset):
    results = []
    for item in tqdm(dataset, desc="HaMI★"):
        q = item.get("question", "")
        pos = item.get("pos") or item.get("gold") or item.get("answer")
        neg = item.get("neg") or item.get("answer2")
        if not pos or not neg:
            continue
        try:
            score_true = calculate_temperature_scaled_agreement(model, q, pos, neg, T=1.5)
            score_false = 1 - score_true
        except Exception as e:
            print(f"[HaMI★] Error: {e}")
            continue
        results.append({"label": 1, "score": score_true})
        results.append({"label": 0, "score": score_false})
    return results
