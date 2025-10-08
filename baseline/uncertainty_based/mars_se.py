import numpy as np
from sentence_transformers import SentenceTransformer, util

def evaluate(model, dataset):
    """
    Baseline: MARS-SE
    Extends MARS by computing similarity entropy over multiple samples.
    """
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    results = []

    for item in dataset:
        q = item.get("question", "")
        pos = item.get("gold") or item.get("pos")
        neg = item.get("neg") or item.get("answer2")

        if not pos or not neg:
            continue

        q_emb = embedder.encode(q, convert_to_tensor=True)
        pos_emb = embedder.encode(pos, convert_to_tensor=True)
        neg_emb = embedder.encode(neg, convert_to_tensor=True)

        sim_pos = float(util.cos_sim(q_emb, pos_emb))
        sim_neg = float(util.cos_sim(q_emb, neg_emb))

        entropy = abs(sim_pos - sim_neg)
        results.append({"label": 1, "score": sim_pos - entropy})
        results.append({"label": 0, "score": sim_neg - entropy})

    return results
