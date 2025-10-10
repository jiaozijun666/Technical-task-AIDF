import os
import json
import subprocess
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



def load_local_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset {path} not found.")
    with open(path, "r") as f:
        data = json.load(f)
    return data

def generate_data_if_needed():
    data_path = "data/quadru.pairs.json"
    if not os.path.exists(data_path):
        print("[INFO] quadru.pairs.json not found. Generating data automatically...")
        subprocess.run(["python", "src/random_pairs.py"], check=True)
        subprocess.run(["python", "src/final_select.py"], check=True)

def compute_auroc(results):
    labels = [r["label"] for r in results if "label" in r and "score" in r]
    scores = [r["score"] for r in results if "label" in r and "score" in r]
    if len(set(labels)) < 2:
        print("[WARN] Only one label present, AUROC cannot be computed.")
        return None
    return roc_auc_score(labels, scores)

def main():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    backend = "hf"
    dataset_path = "data/quadru.pairs.json"

    generate_data_if_needed()
    if os.path.exists(dataset_path):
        dataset = load_local_dataset(dataset_path)
    else:
        dataset = load_dataset("json", data_files=dataset_path)["train"]

    model = get_model(model_id, backend)
    results_summary = {}

    baselines = {
        "p_true": p_true_eval,
        "perplexity": perplexity_eval,
        "semantic_entropy": entropy_eval,
        "mars": mars_eval,
        "mars_se": mars_se_eval,
        "ccs": ccs_eval,
        "saplma": saplma_eval,
        "haloscope": haloscope_eval,
        "HaMI": hami_eval,
        "HaMI_star": hami_star_eval
    }

    for name, func in baselines.items():
        print(f"[INFO] Running baseline: {name}")
        try:
            results = func(model, dataset)
            auroc = compute_auroc(results)
            results_summary[name] = auroc if auroc is not None else "FAILED"
            print(f"[RESULT] {name}: AUROC = {results_summary[name]}")
        except Exception as e:
            print(f"[ERROR] {name} failed: {e}")
            results_summary[name] = "FAILED"

    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", f"auroc_summary.json")
    with open(output_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\n================ Final AUROC Summary ================")
    for k, v in results_summary.items():
        print(f"{k:15s}: {v}")
    print("=====================================================")

if __name__ == "__main__":
    main()
