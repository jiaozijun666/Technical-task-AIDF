import numpy as np
from tqdm import tqdm

def compute_mars(client, data, temperature=0.5):
    results = []
    for item in tqdm(data, desc="MARS"):
        q = item["question"]
        gold = item["gold"]
        neg = item["neg"]
        pos_prompt = f"Question: {q}\nAnswer: {gold}\nRate the factual correctness of this answer from 0 to 1."
        neg_prompt = f"Question: {q}\nAnswer: {neg}\nRate the factual correctness of this answer from 0 to 1."
        pos_score = float(client.text_generation(pos_prompt, model="meta-llama/Llama-3.1-8B",
                                                 temperature=temperature, max_new_tokens=4).strip()[:3])
        neg_score = float(client.text_generation(neg_prompt, model="meta-llama/Llama-3.1-8B",
                                                 temperature=temperature, max_new_tokens=4).strip()[:3])
        results.append(pos_score - neg_score)
    return {"mean": np.mean(results), "scores": results}
