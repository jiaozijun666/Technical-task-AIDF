import os
import json
from src.data import load_json
from baseline.uncertainty_based.p_true import compute_ptrue
from baseline.uncertainty_based.perplexity import compute_perplexity
from baseline.uncertainty_based.semantic_entropy import compute_semantic_entropy
from baseline.uncertainty_based.mars import compute_mars
from baseline.uncertainty_based.mars_se import compute_mars_se
from baseline.internal_representation_based.ccs import compute_ccs
from baseline.internal_representation_based.saplpma import compute_saplma
from baseline.internal_representation_based.haloscope import compute_haloscope
from HaMI.hami import run_hami
from HaMI.enhanced_hami import run_hami_star

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

DATA_PATH = "data/squad_final.json"
OUTPUT_PATH = os.path.join(RESULT_DIR, "final_results.json")

def run_all_baselines():
    print("\n=== Running All Baselines for HaMI Reproduction ===\n")

    results = {}

    # --- Uncertainty-based baselines ---
    print("p(True) baseline ...")
    results["p_true"] = compute_ptrue(DATA_PATH)

    print("Perplexity baseline ...")
    results["perplexity"] = compute_perplexity(DATA_PATH)

    print("Semantic Entropy (SE) baseline ...")
    results["semantic_entropy"] = compute_semantic_entropy(DATA_PATH)

    print("MARS baseline ...")
    results["mars"] = compute_mars(DATA_PATH)

    print("MARS-SE baseline ...")
    results["mars_se"] = compute_mars_se(DATA_PATH)

    # --- Internal representation-based baselines ---
    print("CCS baseline ...")
    results["ccs"] = compute_ccs(DATA_PATH)

    print("SAPLMA baseline ...")
    results["saplma"] = compute_saplma(DATA_PATH)

    print("HaloScope baseline ...")
    results["haloscope"] = compute_haloscope(DATA_PATH)

    # --- HaMI methods ---
    print("HaMI baseline ...")
    results["hami"] = run_hami(DATA_PATH)

    print("HaMI* (enhanced) baseline ...")
    results["hami_star"] = run_hami_star(DATA_PATH)

    # save all outputs
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"All baselines finished! Results saved to {OUTPUT_PATH}\n")

if __name__ == "__main__":
    run_all_baselines()
