import numpy as np
from tqdm import tqdm

def compute_mars_se(client, data, n_samples=5, temperature=0.5):
    results = []
    for item in tqdm(data, desc="MARS-SE"):
        q = item["question"]
        gold = item["gold"]
        scores = []
        for _ in range(n_samples):
            prompt = f"Question: {q}\nAnswer: {gold}\nRate the factual correctness of this answer from 0 to 1."
            res = client.text_generation(prompt, model="meta-llama/Llama-3.1-8B",
                                         temperature=temperature, max_new_tokens=4)
            try:
                score = float(res.strip()[:3])
                scores.append(score)
            except:
                continue
        entropy = -np.sum(np.log(np.clip(scores, 1e-8, 1))) / len(scores)
        results.append(entropy)
    return {"mean": np.mean(results), "scores": results}
