import numpy as np
from tqdm import tqdm

def compute_saplma(client, data, temperature=0.5):
    results = []
    for item in tqdm(data, desc="SAPLMA"):
        q = item["question"]
        gold = item["gold"]
        prompt = f"How semantically aligned is this answer to the question (0-1)?\nQ: {q}\nA: {gold}"
        res = client.text_generation(prompt, model="meta-llama/Llama-3.1-8B",
                                     temperature=temperature, max_new_tokens=4)
        try:
            score = float(res.strip()[:3])
        except:
            score = 0.0
        results.append(score)
    return {"mean": np.mean(results), "scores": results}
