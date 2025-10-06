import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def compute_mars(data_path, output_path="results/mars.jsonl", model_name="all-MiniLM-L6-v2"):
    """
    MARS: Mean Agreement of Response Similarity
    Measures average semantic agreement among generations.
    """
    os.makedirs("results", exist_ok=True)
    data = json.load(open(data_path))
    model = SentenceTransformer(model_name)

    results = []
    for item in tqdm(data, desc="Computing MARS"):
        gens = item.get("generations", [])
        if len(gens) < 2:
            continue

        embeddings = model.encode(gens, normalize_embeddings=True)
        sim_matrix = np.dot(embeddings, embeddings.T)
        np.fill_diagonal(sim_matrix, 0) 
        mars_score = float(np.mean(sim_matrix))  

        results.append({
            "question": item["question"],
            "score": mars_score
        })

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[✔] Saved MARS results to {output_path}")
    return results
