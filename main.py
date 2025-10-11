import os
import numpy as np
import json
import subprocess
from huggingface_hub import InferenceClient
from baseline.uncertainty_based.p_true import compute_p_true
from baseline.uncertainty_based.perplexity import compute_perplexity
from baseline.uncertainty_based.semantic_entropy import compute_semantic_entropy
from baseline.uncertainty_based.mars import compute_mars
from baseline.uncertainty_based.mars_se import compute_mars_se
from baseline.internal_representation_based.ccs import compute_ccs
from baseline.internal_representation_based.saplma import compute_saplma
from baseline.internal_representation_based.haloscope import compute_haloscope
from HaMI.hami import compute_hami
from HaMI.hami_star import compute_hami_star
from sklearn.metrics import roc_auc_score, accuracy_score


def generate_data():
    os.makedirs("data", exist_ok=True)
    subprocess.run(["python", "process_data.py"], check=True)
    subprocess.run(["python", "random_pairs.py"], check=True)
    subprocess.run(["python", "multi_sample.py"], check=True)
    subprocess.run(["python", "refined_set.py"], check=True)

def load_refined():
    with open("data/squad_refined.json") as f:
        return json.load(f)

def run_all(client, data):
    results = {}
    results["p_true"] = compute_p_true(client, data)
    results["perplexity"] = compute_perplexity(client, data)
    results["semantic_entropy"] = compute_semantic_entropy(client, data)
    results["mars"] = compute_mars(client, data)
    results["mars_se"] = compute_mars_se(client, data)
    results["ccs"] = compute_ccs(client, data)
    results["saplma"] = compute_saplma(client, data)
    results["haloscope"] = compute_haloscope(client, data)
    results["hami"] = compute_hami(client, data)
    results["hami_star"] = compute_hami_star(client, data)
    return results

def evaluate(results, data):
    y_true = [1 for _ in data]
    metrics = {}
    for name, res in results.items():
        y_pred = np.array(res["scores"])
        if np.all(y_pred == y_pred[0]):
            auc = 0.5
        else:
            auc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(np.round(y_pred), y_true)
        metrics[name] = {"AUROC": float(auc), "ACC": float(acc)}
    return metrics

def main():
    if not os.path.exists("data/squad_refined.json"):
        generate_data()
    data = load_refined()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("Missing Hugging Face token.")
    client = InferenceClient(model="meta-llama/Llama-3.1-8B", token=token)
    os.makedirs("results", exist_ok=True)
    results = run_all(client, data)
    metrics = evaluate(results, data)
    json.dump({"scores": results, "metrics": metrics}, open("results/final_results.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()