import numpy as np
from tqdm import tqdm

def compute_hami(client, data, temperature=0.3):
    results = []
    for item in tqdm(data, desc="HaMI"):
        q = item["question"]
        gold = item["gold"]
        neg = item["neg"]
        prompt = f"Which answer is more factually correct for the question?\nQuestion: {q}\nA: {gold}\nB: {neg}\nAnswer A or B."
        res = client.text_generation(prompt, model="meta-llama/Llama-3.1-8B",
                                     temperature=temperature, max_new_tokens=4)
        pred = 1 if "a" in res.lower() else 0
        results.append(pred)
    return {"mean": np.mean(results), "scores": results}
