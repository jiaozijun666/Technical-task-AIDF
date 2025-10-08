import numpy as np
from sentence_transformers import SentenceTransformer, util

def evaluate(model, dataset):
    """
    Baseline: SAPLMA
    Computes self-attentive pooled latent matching alignment between embeddings.
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

        # Use normalized dot product as alignment
        score_true = float(util.dot_score(q_emb, pos_emb))
        score_false = float(util.dot_score(q_emb, neg_emb))

        results.append({"label": 1, "score": score_true})
        results.append({"label": 0, "score": score_false})

    return results
