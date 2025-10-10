import random
import json
import os

def generate_random_pairs(num_samples: int = 200):
    data = []
    for i in range(num_samples):
        q = f"Question {i+1}: What is the meaning of sample {i+1}?"
        pos = f"Answer {i+1}: This is a correct answer to sample {i+1}."
        neg = f"Answer {i+1}: This is an incorrect answer to sample {i+1}."
        data.append({"question": q, "pos": pos, "neg": neg})
    os.makedirs("data", exist_ok=True)
    path = "data/squad_random_pairs.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Generated {len(data)} pairs → {path}")
    return data
