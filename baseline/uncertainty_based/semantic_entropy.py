# baseline/uncertainty_based/semantic_entropy.py
import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def compute_semantic_entropy(data_path, output_path="results/semantic_entropy.jsonl", model_name="all-MiniLM-L6-v2"):
    """
    Semantic Entropy baseline:
    Measures disagreement among multiple generations in embedding space.
    """
    os.makedirs("results", exist_ok=True)
    data = json.load(open(data_path))
    model = SentenceTransformer(model_name)

    results = []
    for item in tqdm(data, desc="Computing Semantic Entropy"):
        gens = item.get("generations", [])
        if len(gens) < 2:
            continue

        # 计算句向量并归一化
        embeddings = model.encode(gens, normalize_embeddings=True)
        # 计算两两余弦相似度矩阵
        sim = np.dot(embeddings, embeddings.T)
        # 取上三角平均相似度
        avg_sim = np.mean(sim[np.triu_indices_from(sim, k=1)])
        sem_entropy = float(1 - avg_sim)

        results.append({
            "question": item["question"],
            "score": sem_entropy
        })

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[✔] Saved semantic entropy results to {output_path}")
    return results
