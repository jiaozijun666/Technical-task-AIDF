import numpy as np
from tqdm import tqdm

def compute_semantic_entropy(client, data, n_samples=5, temperature=0.5):
    results = []
    for item in tqdm(data, desc="Semantic Entropy"):
        q = item["question"]
        gold = item["gold"]
        generations = []
        for _ in range(n_samples):
            res = client.text_generation(f"Question: {q}\nAnswer:", 
                                         model="meta-llama/Llama-3.1-8B",
                                         temperature=temperature, max_new_tokens=64)
            generations.append(res.strip())
        unique_ratio = len(set(generations)) / n_samples
        entropy = -np.log(unique_ratio + 1e-8)
        results.append(entropy)
    return {"mean": np.mean(results), "scores": results}
