import numpy as np
from sentence_transformers import SentenceTransformer, util

def evaluate(model, dataset):
    """
    Baseline: CCS
    Computes contextual consistency score between question and answer embeddings.
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

        score_true = float(util.cos_sim(q_emb, pos_emb))
        score_false = float(util.cos_sim(q_emb, neg_emb))

        results.append({"label": 1, "score": score_true})
        results.append({"label": 0, "score": score_false})

    return results
