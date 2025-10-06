# src/random_pairs.py
import os
import json
import random
from tqdm import tqdm

def create_random_pairs(input_path="data/squad_final.json", 
                        output_path="data/squad_random_pairs.json", 
                        seed=42):
    """
    Create random mismatched (question, answer) pairs 
    for negative examples (hallucinations).
    """
    random.seed(seed)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found.")

    with open(input_path, "r") as f:
        data = json.load(f)

    # 取出所有正确答案
    all_answers = [item["gold"] for item in data]
    n = len(data)
    results = []

    print(f"Generating {n} random mismatched QA pairs...")
    for item in tqdm(data):
        q = item["question"]
        # 随机抽取不同问题的答案
        wrong = item["gold"]
        while wrong == item["gold"]:
            wrong = random.choice(all_answers)
        results.append({
            "question": q,
            "gold": item["gold"],
            "random_answer": wrong,
            "label": 0
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✔] Saved {len(results)} random pairs to {output_path}")
    return results

if __name__ == "__main__":
    create_random_pairs()
