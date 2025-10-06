# baseline/internal_representation_based/haloscope.py
import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from prompt import get_generation_prompt  # 可统一调用 prompt（保持一致性）

def compute_haloscope(data_path, output_path="results/haloscope.jsonl", model_name="all-MiniLM-L6-v2"):
    """
    HALOSCOPE baseline:
    Measures cosine similarity between generation and gold answer embeddings.
    """
    os.makedirs("results", exist_ok=True)
    data = json.load(open(data_path))
    encoder = SentenceTransformer(model_name)

    results = []
    for item in tqdm(data, desc="Computing HALOSCOPE"):
        q = item["question"]
        gold = item.get("gold", "")
        gens = item.get("generations", [])
        if not gens or not gold:
            continue

        # Reference embedding
        ref_emb = encoder.encode(gold, normalize_embeddings=True)

        for g in gens:
            prompt = get_generation_prompt(q)  # 用统一生成风格
            gen_emb = encoder.encode(g, normalize_embeddings=True)
            score = float(np.dot(ref_emb, gen_emb))  # 余弦相似度
            results.append({
                "question": q,
                "gold": gold,
                "generation": g,
                "score": score
            })

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[✔] Saved HALOSCOPE results to {output_path}")
    return results
