import numpy as np
from tqdm import tqdm

def compute_ccs(client, data, temperature=0.5):
    results = []
    for item in tqdm(data, desc="CCS"):
        q = item["question"]
        gold = item["gold"]
        prompt = f"Does this answer logically follow from the question?\nQ: {q}\nA: {gold}\nAnswer Yes or No."
        res = client.text_generation(prompt, model="meta-llama/Llama-3.1-8B",
                                     temperature=temperature, max_new_tokens=4)
        score = 1.0 if "yes" in res.lower() else 0.0
        results.append(score)
    return {"mean": np.mean(results), "scores": results}
