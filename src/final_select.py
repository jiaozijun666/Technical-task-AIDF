import os
import json
import random

def load_multi_sample(path="data/squad_multi.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run multi_sample.py first.")
    with open(path, "r") as f:
        data = json.load(f)
    return data

def filter_generations(generations, min_len=10):
    filtered = []
    for g in generations:
        text = g.strip()
        if len(text) < min_len:     
            continue
        if any(x in text.lower() for x in ["sorry", "as an ai", "language model"]):
            continue
        filtered.append(text)
    return filtered

def select_high_quality(data, sample_size=500, retain_size=400, seed=42, output_path = "data/quadru_pairs.json"):
    random.seed(seed)
    sampled = random.sample(data, min(sample_size, len(data)))
    final_data = []

    for item in sampled:
        gens = filter_generations(item["generations"])
        if len(gens) == 0:
            continue
        final_data.append({
            "question": item["question"],
            "gold": item["gold"],
            "generations": gens
        })

    if len(final_data) > retain_size:
        final_data = random.sample(final_data, retain_size)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_data, f, indent=2)

    print(f"[✔] Selected {len(final_data)} clean samples saved to {output_path}")

if __name__ == "__main__":
    data = load_multi_sample("data/squad_multi.json")  
    select_high_quality(data)
