import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

def compute_logprob(model, tok, prompt, completion):
    """Compute average log-probability of completion given prompt."""
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    labels = tok(completion, return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
    return -loss.item()  # negative log-likelihood

def hami_score(models, tok, q, a_pos, a_neg):
    """Base HaMI score = agreement among models"""
    probs = []
    for m in models:
        p_pos = np.exp(compute_logprob(m, tok, q, a_pos))
        p_neg = np.exp(compute_logprob(m, tok, q, a_neg))
        probs.append(p_pos / (p_pos + p_neg + 1e-9))
    return float(np.mean(probs))

def hami_star_score(models, tok, q, a_pos, a_neg, T=1.5):
    """HaMI★ = temperature-scaled log-prob agreement"""
    log_ratios = []
    for m in models:
        lp_pos = compute_logprob(m, tok, q, a_pos)
        lp_neg = compute_logprob(m, tok, q, a_neg)
        log_ratios.append((lp_pos - lp_neg) / T)
    probs = 1 / (1 + np.exp(-np.array(log_ratios)))
    return float(np.mean(probs))

def run_all_baselines(model, data_path):
    data = json.load(open(data_path))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model.model_id)

    print("[INFO] Running MARS, CCS, and HaMI baselines...")
    results = {"MARS": [], "CCS": [], "HaMI": [], "HaMI_star": []}

    # replicate 3-model ensemble (paper: Section 5.1)
    models = [model.mdl for _ in range(3)]

    for item in tqdm(data[:100]):
        q = item["prompt"]
        a_pos = item["response"]
        a_neg = item["negative"]

        # MARS (semantic similarity baseline)
        mars = len(set(a_pos.split()) & set(a_neg.split())) / (len(set(a_pos.split()) | set(a_neg.split())) + 1e-9)
        results["MARS"].append(mars)

        # CCS (contextual consistency score)
        ccs = 1 - abs(len(a_pos) - len(a_neg)) / max(len(a_pos), len(a_neg))
        results["CCS"].append(ccs)

        # HaMI / HaMI★
        results["HaMI"].append(hami_score(models, tok, q, a_pos, a_neg))
        results["HaMI_star"].append(hami_star_score(models, tok, q, a_pos, a_neg))

    print("[INFO] Baselines complete.")
    return results
