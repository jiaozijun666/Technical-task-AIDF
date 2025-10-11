import numpy as np
from tqdm import tqdm

def compute_p_true(client, data, temperature=0.5):
    results = []
    for item in tqdm(data, desc="p_true"):
        q = item["question"]
        gold = item["gold"]
        prompt = f"Question: {q}\nAnswer:"
        res = client.text_generation(prompt, model="meta-llama/Llama-3.1-8B",
                                     temperature=temperature, max_new_tokens=64)
        score = float(res.lower().strip() in gold.lower())
        results.append(score)
    return {"mean": np.mean(results), "scores": results}
