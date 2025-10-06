# baseline/uncertainty_based/mars_se.py
import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def compute_mars_se(data_path, output_path="results/mars_se.jsonl", model_name="all-MiniLM-L6-v2"):
    """
    MARS-SE: Mean Agreement of Response Similarity with Entropy weighting.
    Adds Shannon entropy weighting to MARS for more stable uncertainty estimates.
    """
    os.makedirs("results", exist_ok=True)
    data = json.load(open(data_path))
    model = SentenceTransformer(model_name)

    results = []
    for item in tqdm(data, desc="Computing MARS-SE"):
        gens = item.get("generations", [])
        if len(gens) < 2:
            continue

        embeddings = model.encode(gens, normalize_embeddings=True)
        sim_matrix = np.dot(embeddings, embeddings.T)
        np.fill_diagonal(sim_matrix, 0)

        # 计算每个样本的平均相似度
        avg_sim = sim_matrix.mean(axis=1)
        # 归一化后计算熵权重
        p = avg_sim / (np.sum(avg_sim) + 1e-8)
        entropy = -np.sum(p * np.log(p + 1e-8))  # Shannon entropy
        entropy_weight = 1 / (1 + entropy)       # 熵越大，不确定性越高 → 权重越小

        mars_se_score = float(np.mean(avg_sim) * entropy_weight)
        results.append({
            "question": item["question"],
            "score": mars_se_score
        })

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[✔] Saved MARS-SE results to {output_path}")
    return results
