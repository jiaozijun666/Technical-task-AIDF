import numpy as np
from tqdm import tqdm

def compute_hami_star(client, data, temperature=0.3):
    results = []
    for item in tqdm(data, desc="HaMI*"):
        q = item["question"]
        gold = item["gold"]
        neg = item["neg"]
        prompt = f"Rate the factual correctness difference between A and B on a 0-1 scale.\nQ: {q}\nA: {gold}\nB: {neg}"
        res = client.text_generation(prompt, model="meta-llama/Llama-3.1-8B",
                                     temperature=temperature, max_new_tokens=6)
        try:
            score = float(res.strip()[:4])
        except:
            score = 0.5
        results.append(score)
    return {"mean": np.mean(results), "scores": results}
