import os
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score
from src.model import get_model
from src.data import load_dataset
from src.random_pairs import generate_random_pairs
from baseline.uncertainty_based.p_true import evaluate as eval_p_true
from baseline.uncertainty_based.perplexity import evaluate as eval_perplexity
from baseline.uncertainty_based.semantic_entropy import evaluate as eval_semantic_entropy
from baseline.uncertainty_based.mars import evaluate as eval_mars
from baseline.uncertainty_based.mars_se import evaluate as eval_mars_se
from baseline.internal_representation_based.ccs import evaluate as eval_ccs
from baseline.internal_representation_based.saplma import evaluate as eval_saplma
from baseline.internal_representation_based.haloscope import evaluate as eval_haloscope
from HaMI.hami import run_hami
from HaMI.hami_star import run_hami_star


def evaluate_baseline(eval_fn, model, dataset, name):
    print(f"\n[INFO] Running baseline: {name}")
    results = eval_fn(model, dataset)
    y_true = [r["label"] for r in results if "label" in r]
    y_score = [r["score"] for r in results if "score" in r]

    if len(set(y_true)) < 2:
        print(f"[WARN] {name}: Only one label present, AUROC cannot be computed.")
        return None

    auc = roc_auc_score(y_true, y_score)
    print(f"[RESULT] {name}: AUROC = {auc:.4f}")
    return auc

def main():
    # Step 1. Model setup
    model_id = "meta-llama/Llama-3.2-8B-Instruct"
    print(f"[INFO] Initializing model: {model_id}")
    model = get_model(model_id)
    print("[INFO] Model loaded successfully.")

    # Step 2. Dataset preparation
    data_path = "data/squad_random_pairs.json"
    if not os.path.exists(data_path):
        print("[WARN] Dataset not found. Generating random pairs...")
        dataset = generate_random_pairs()
    else:
        dataset = load_dataset(data_path)

    print(f"[INFO] Dataset loaded: {len(dataset)} samples")

    # Step 3. Define all baselines
    baselines = {
        # Uncertainty-based
        "p_true": eval_p_true,
        "perplexity": eval_perplexity,
        "semantic_entropy": eval_semantic_entropy,
        "mars": eval_mars,
        "mars_se": eval_mars_se,

        # Internal representation-based
        "ccs": eval_ccs,
        "saplma": eval_saplma,
        "haloscope": eval_haloscope,

        # HaMI variants
        "HaMI": run_hami,
        "HaMI_star": run_hami_star,
    }

    # Step 4. Evaluate all baselines
    results = {}
    for name, fn in baselines.items():
        try:
            auc = evaluate_baseline(fn, model, dataset, name)
            results[name] = auc
        except Exception as e:
            print(f"[ERROR] {name} failed: {e}")
            results[name] = None

    # Step 5. Save summary results
    os.makedirs("results", exist_ok=True)
    result_path = f"results/auroc_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] AUROC summary saved to {result_path}")

    # Step 6. Print summary table
    print("\n==================== Final AUROC Summary ====================")
    for name, auc in results.items():
        if auc is None:
            print(f"{name:<20} : FAILED")
        else:
            print(f"{name:<20} : {auc:.4f}")
    print("=============================================================\n")


if __name__ == "__main__":
    main()