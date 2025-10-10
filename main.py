import os
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score
from src.model import get_model
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


def compute_auroc(preds):
    """Compute AUROC given a list of {'label': int, 'score': float}"""
    labels = [p["label"] for p in preds]
    scores = [p["score"] for p in preds]
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def print_summary_table(results_dict):
    """Print AUROC summary in a clean table format"""
    print("\n" + "=" * 60)
    print("AUROC SUMMARY TABLE")
    print("=" * 60)
    for k, v in results_dict.items():
        if isinstance(v, float):
            print(f"{k:<20} {v:.3f}")
        else:
            print(f"{k:<20} {v}")
    print("=" * 60 + "\n")


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def main():
    print("\nStarting full benchmark run...\n")

    # Step 1: Prepare dataset
    print("[INFO] Step 1: Preparing dataset...")
    data_path = "data/squad_random_pairs.json"
    if not os.path.exists(data_path):
        dataset = generate_random_pairs(num_samples=200)
    else:
        with open(data_path, "r") as f:
            dataset = json.load(f)
        print(f"[INFO] Loaded existing dataset with {len(dataset)} samples.")

    # Step 2: Load model
    print("\n[INFO] Step 2: Loading model...")
    model = get_model("meta-llama/Llama-3.1-8B-Instruct", backend="hf")

    # Step 3: Run all baselines
    print("\n[INFO] Step 3: Running baseline models...")
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
    }

    results = {}
    for name, func in baselines.items():
        print(f"[INFO] Running {name} ...")
        preds = func(model, dataset)
        auroc = compute_auroc(preds)
        results[name] = auroc
        print(f"{name} AUROC = {auroc:.3f}")

    # Step 4: Run HaMI and HaMI★
    print("\n[INFO] Step 4: Running HaMI and HaMI★ ...")
    results["HaMI"] = run_hami(model, dataset)
    results["HaMI★"] = run_hami_star(model, dataset)

    # Step 5: Summarize and save
    print_summary_table(results)
    ensure_dir("results")
    save_path = os.path.join("results", "summary.json")
    results["timestamp"] = datetime.now().isoformat()
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {save_path}\n")


if __name__ == "__main__":
    main()
