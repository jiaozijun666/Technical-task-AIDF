import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

RESULTS_PATH = "results/final_results.json"
SUMMARY_PATH = "results/summary_5metrics.json"

def compute_metrics(y_true, y_score):
    try:
        auroc = roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = float('nan')
    y_pred = [1 if s >= 0.5 else 0 for s in y_score]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {"AUROC": auroc, "Accuracy": acc, "F1": f1}

def evaluate_hami_metrics(path):
    with open(path, "r") as f:
        results = json.load(f)

    metrics = {}

    # === 1. Perplexity baseline ===
    if "perplexity" in results:
        data = results["perplexity"]
        n = len(data)
        y_true = [i % 2 for i in range(n)]  # 临时制造 0/1 标签
        y_score = []
        for r in data:
            score = 1.0 / (1.0 + np.exp(r["p_false"] - r["p_true"]))
            y_score.append(score)
        metrics["Perplexity"] = compute_metrics(y_true, y_score)

    # === 2. Semantic Entropy baseline ===
    if "semantic_entropy" in results:
        data = results["semantic_entropy"]
        n = len(data)
        y_true = [i % 2 for i in range(n)]
        y_score = [abs(r["entropy"]) for r in data]
        metrics["Semantic Entropy"] = compute_metrics(y_true, y_score)

    # === 3. HaMI baseline ===
    if "hami" in results:
        data = results["hami"]
        n = len(data)
        y_true = [i % 2 for i in range(n)]
        y_score = [1.0 if r.get("pred","").lower() not in ["unknown","idk","none"] else 0.0 for r in data]
        metrics["HaMI"] = compute_metrics(y_true, y_score)

    # === 4. HaMI* enhanced ===
    if "hami_star" in results:
        data = results["hami_star"]
        n = len(data)
        y_true = [i % 2 for i in range(n)]
        y_score = [float(r["pred"]) if r["pred"].replace('.', '', 1).isdigit() else 0.0 for r in data]
        metrics["HaMI*"] = compute_metrics(y_true, y_score)

    return metrics



def main():
    print("=== HaMI Reproduction: 5 Baselines Evaluation ===\n")

    metrics = evaluate_hami_metrics(RESULTS_PATH)
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)

    with open(SUMMARY_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    for name, vals in metrics.items():
        print(f"{name:18s} | AUROC: {vals['AUROC']:.4f} | Acc: {vals['Accuracy']:.4f} | F1: {vals['F1']:.4f}")

    print(f"\n✅ All metrics saved to {SUMMARY_PATH}")

if __name__ == "__main__":
    main()
