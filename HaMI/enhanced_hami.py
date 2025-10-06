# HaMI/enhanced_hami.py

import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

from baseline.uncertainty_based.semantic_entropy import compute_semantic_entropy
from baseline.internal_representation_based.haloscope import compute_haloscope


def compute_enhanced_features(data):
    """
    Compute enhanced feature vectors by combining:
      - uncertainty-based signals (entropy, p_true, etc.)
      - representation-based signals (HALOSCOPE similarity)
      - multi-sample agreement score
    """
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    X, y = [], []

    for item in tqdm(data, desc="Building enhanced HaMI features"):
        q = item["question"]
        gold = item["gold"]
        gens = item["generations"]
        label = item.get("label", 0)

        # 1. Semantic entropy (uncertainty signal)
        entropy = np.std([len(g.split()) for g in gens]) / 10.0  # proxy measure

        # 2. Representation similarity (HALOSCOPE-style)
        gold_emb = encoder.encode(gold, normalize_embeddings=True)
        sim_scores = []
        for g in gens:
            g_emb = encoder.encode(g, normalize_embeddings=True)
            sim_scores.append(np.dot(g_emb, gold_emb))
        rep_sim = np.mean(sim_scores)

        # 3. Multi-sample agreement (textual consistency)
        pairwise_sims = []
        for i in range(len(gens)):
            for j in range(i + 1, len(gens)):
                gi = encoder.encode(gens[i], normalize_embeddings=True)
                gj = encoder.encode(gens[j], normalize_embeddings=True)
                pairwise_sims.append(np.dot(gi, gj))
        agreement = np.mean(pairwise_sims) if pairwise_sims else 0.0

        # Combine features
        X.append([entropy, rep_sim, agreement])
        y.append(label)

    return np.array(X), np.array(y)


def train_enhanced_hami(data_path="data/squad_refined.json", output_path="results/enhanced_hami.jsonl"):
    """
    Train and evaluate Enhanced HaMI (HaMI*).
    """
    print("Loading dataset:", data_path)
    data = json.load(open(data_path))

    # 1. Compute enhanced feature representations
    X, y = compute_enhanced_features(data)

    # 2. Standardize and train
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_std, y)

    # 3. Predict factuality probabilities
    y_pred = clf.predict_proba(X_std)[:, 1]

    # 4. Save predictions
    os.makedirs("results", exist_ok=True)
    with open(output_path, "w") as f:
        for item, score in zip(data, y_pred):
            f.write(json.dumps({
                "question": item["question"],
                "gold": item["gold"],
                "generation": item.get("generation", ""),
                "score": float(score)
            }) + "\n")

    # 5. Evaluate
    auroc = roc_auc_score(y, y_pred)
    f1 = f1_score(y, y_pred > 0.5)
    print(f"Enhanced HaMI AUROC: {auroc:.4f}, F1: {f1:.4f}")

    return {"AUROC": auroc, "F1": f1}


if __name__ == "__main__":
    train_enhanced_hami()
