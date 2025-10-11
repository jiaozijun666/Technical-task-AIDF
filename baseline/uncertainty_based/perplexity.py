import numpy as np
from tqdm import tqdm

def compute_perplexity(client, data, temperature=0.5):
    results = []
    for item in tqdm(data, desc="Perplexity"):
        q = item["question"]
        gold = item["gold"]
        neg = item["neg"]
        prompt_true = f"Question: {q}\nAnswer: {gold}"
        prompt_false = f"Question: {q}\nAnswer: {neg}"
        res_true = client.text_generation(prompt_true, model="meta-llama/Llama-3.1-8B",
                                          temperature=temperature, max_new_tokens=1)
        res_false = client.text_generation(prompt_false, model="meta-llama/Llama-3.1-8B",
                                           temperature=temperature, max_new_tokens=1)
        ppl = np.exp(abs(len(res_true) - len(res_false)))
        results.append(ppl)
    return {"mean": np.mean(results), "scores": results}
