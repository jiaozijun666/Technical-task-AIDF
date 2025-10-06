import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support

from baseline.uncertainty_based.perplexity import compute_perplexity
from baseline.uncertainty_based.semantic_entropy import compute_semantic_entropy
from baseline.uncertainty_based.mars import compute_mars
from baseline.uncertainty_based.mars_se import compute_mars_se
from baseline.uncertainty_based.p_true import compute_p_true

from baseline.internal_representation_based.haloscope import compute_haloscope
from baseline.internal_representation_based.ccs import compute_ccs
from baseline.internal_representation_based.saplma import compute_saplma

from HaMI.hami import train_and_evaluate_hami
from HaMI.enhanced_hami import train_enhanced_hami


RESULT_DIR = "results"


def evaluate_predictions(pred_path, gold_path="data/squad_refined.json"):
    """
    Compute AUROC, F1, Precision, Recall for a given baseline output.
    """
    if not os.path.exists(pred_path):
        print(f"Warning: {pred_path} not found, skipping.")
        return None

    with open(gold_path, "r") as f:
        gold_data = json.load(f)
    gold_dict = {(d["question"], d.get("generation", "")): d["label"] for d in gold_data}

    preds, labels = [], []
    with open(pred_path, "r") as f:
        for line in f:
            d = json.loads(line)
            key = (d["question"], d.get("generation", ""))
            if key in gold_dict:
                preds.append(d["score"])
                labels.append(gold_dict[key])

    if len(labels) < 2:
        print(f"Warning: Not enough valid samples in {pred_path}")
        return None

    preds, labels = np.array(preds), np.array(labels)
    auroc = roc_auc_score(labels, preds)
    f1 = f1_score(labels, preds > 0.5)
    prec, rec, _, _ = precision_recall_fscore_support(labels, preds > 0.5, average="binary")

    return {
        "AUROC": round(auroc, 4),
        "F1": round(f1, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "N": len(labels)
    }


# ============================================================
# Main pipeline
# ============================================================
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    data_path = "data/squad_refined.json"
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    print("Running all baselines and HaMI variants...\n")

    # === Step 1: Run all baselines ===
    compute_perplexity(model_name, data_path, f"{RESULT_DIR}/perplexity.jsonl")
    compute_semantic_entropy(data_path, f"{RESULT_DIR}/semantic_entropy.jsonl")
    compute_mars(data_path, f"{RESULT_DIR}/mars.jsonl")
    compute_mars_se(data_path, f"{RESULT_DIR}/mars_se.jsonl")
    compute_p_true(model_name, data_path, f"{RESULT_DIR}/p_true.jsonl")

    compute_haloscope(data_path, f"{RESULT_DIR}/haloscope.jsonl")
    compute_ccs(model_name, data_path, f"{RESULT_DIR}/ccs.jsonl")
    compute_saplma(model_name, data_path, f"{RESULT_DIR}/saplma.jsonl")

    # === Step 2: Run HaMI (basic) ===
    train_and_evaluate_hami(data_path, model_name, f"{RESULT_DIR}/hami.jsonl")

    # === Step 3: Run Enhanced HaMI (HaMI*) ===
    train_enhanced_hami(data_path, f"{RESULT_DIR}/enhanced_hami.jsonl")

    # === Step 4: Evaluate and summarize ===
    baselines = [
        "perplexity",
        "semantic_entropy",
        "mars",
        "mars_se",
        "p_true",
        "haloscope",
        "ccs",
        "saplma",
        "hami",
        "enhanced_hami"
    ]

    summary = {}
    for b in tqdm(baselines, desc="Evaluating all methods"):
        pred_file = os.path.join(RESULT_DIR, f"{b}.jsonl")
        metrics = evaluate_predictions(pred_file, data_path)
        if metrics:
            summary[b] = metrics

    # === Step 5: Save summary ===
    summary_path = os.path.join(RESULT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Evaluation Summary ===")
    for name, m in summary.items():
        print(f"{name:<18} | AUROC: {m['AUROC']:.4f} | F1: {m['F1']:.4f} | "
              f"P: {m['Precision']:.4f} | R: {m['Recall']:.4f}")
    print(f"\nSummary saved to {summary_path}")



if __name__ == "__main__":
    main()
