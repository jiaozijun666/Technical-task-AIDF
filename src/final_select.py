# src/final_select.py
import os
import json
import random

def load_multi_sample(path="data/squad_multi.json"):
    """加载 multi_sample 生成的结果"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run multi_sample.py first.")
    with open(path, "r") as f:
        data = json.load(f)
    return data

def filter_generations(generations, min_len=10):
    """过滤低质量回答"""
    filtered = []
    for g in generations:
        text = g.strip()
        if len(text) < min_len:           # 太短
            continue
        if any(x in text.lower() for x in ["sorry", "as an ai", "language model"]):
            continue
        filtered.append(text)
    return filtered

def select_high_quality(data, sample_size=500, retain_size=400, seed=42, output_path="data/squad_final.json"):
    """随机抽样并筛选"""
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

    # 如果筛完后还多于 retain_size，随机取 retain_size 条
    if len(final_data) > retain_size:
        final_data = random.sample(final_data, retain_size)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_data, f, indent=2)

    print(f"[✔] Selected {len(final_data)} clean samples saved to {output_path}")

if __name__ == "__main__":
    data = load_multi_sample("data/squad_multi.json")  # 或改为 squad_multi_debug.json
    select_high_quality(data)
