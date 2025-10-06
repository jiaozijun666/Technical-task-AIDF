import os
import json
import numpy as np
from tqdm import tqdm
from model import get_model, GenConfig
from prompt import (
    build_eval_prompt,
    build_hami_prompt,
    build_hami_star_prompt,
)


def extract_answers(item):
    """
    Automatically detect which keys correspond to the two answers.
    Compatible with different JSON formats from random_pairs.py.
    """
    q = item.get("question", "")
    candidates = list(item.keys())
    ignore = {"question", "id", "index"}
    candidates = [k for k in candidates if k not in ignore]

    if len(candidates) < 2:
        return q, None, None

    # Common naming patterns
    mapping = {
        "pos": ["pos", "positive", "gold", "correct", "answer1"],
        "neg": ["neg", "negative", "wrong", "incorrect", "answer2"],
    }

    a_true, a_false = None, None
    for k in candidates:
        key_lower = k.lower()
        if any(t in key_lower for t in mapping["pos"]):
            a_true = item[k]
        elif any(t in key_lower for t in mapping["neg"]):
            a_false = item[k]

    # fallback
    if not a_true:
        a_true = item[candidates[0]]
    if not a_false and len(candidates) > 1:
        a_false = item[candidates[1]]

    return q, a_true, a_false



def compute_perplexity(model, pairs):
    results = []
    for item in tqdm(pairs, desc="Perplexity baseline"):
        q, a_true, a_false = extract_answers(item)
        if not a_true or not a_false:
            continue

        p_true = model.nll(q, a_true)
        p_false = model.nll(q, a_false)
        results.append({"question": q, "p_true": p_true, "p_false": p_false})
    return results


def compute_semantic_entropy(model, pairs):
    results = []
    for item in tqdm(pairs, desc="Semantic Entropy baseline"):
        q, a_true, a_false = extract_answers(item)
        if not a_true or not a_false:
            continue

        prompt = build_eval_prompt(q, [a_true, a_false])
        cfg = GenConfig(temperature=0.7, top_p=0.9, max_new_tokens=64)
        outs = [model.generate(prompt, cfg).text for _ in range(3)]
        # entropy approximation
        unique_ratio = len(set(outs)) / max(len(outs), 1)
        entropy = -np.log(unique_ratio + 1e-8)
        results.append({"question": q, "entropy": float(entropy)})
    return results


def compute_hami(model, pairs):
    results = []
    for item in tqdm(pairs, desc="HaMI baseline"):
        q, a_true, a_false = extract_answers(item)
        if not a_true or not a_false:
            continue

        prompt = build_hami_prompt(q, a_true, a_false)
        cfg = GenConfig(temperature=0.3, top_p=1.0, max_new_tokens=10)
        out = model.generate(prompt, cfg).text.strip()
        pred = "A" if "A" in out else ("B" if "B" in out else "unknown")
        results.append({"question": q, "pred": pred})
    return results


def compute_hami_star(model, pairs):
    results = []
    for item in tqdm(pairs, desc="HaMI* baseline"):
        q, a_true, a_false = extract_answers(item)
        if not a_true or not a_false:
            continue

        prompt = build_hami_star_prompt(q, a_true, a_false)
        cfg = GenConfig(temperature=0.3, top_p=1.0, max_new_tokens=10)
        out = model.generate(prompt, cfg).text.strip()
        pred = "1" if "1" in out else ("2" if "2" in out else "unknown")
        results.append({"question": q, "pred": pred})
    return results


def main():
    input_path = "data/squad_random_pairs.json"
    output_path = "results/final_results.json"
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found. Run random_pairs.py first.")

    with open(input_path, "r") as f:
        pairs = json.load(f)

    print(f"Loaded {len(pairs)} pairs for evaluation.")
    model = get_model(model_id, backend="hf")

    results = {
        "perplexity": compute_perplexity(model, pairs[:10]),  # subset for speed
        "semantic_entropy": compute_semantic_entropy(model, pairs[:5]),
        "hami": compute_hami(model, pairs[:5]),
        "hami_star": compute_hami_star(model, pairs[:5]),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"aved final evaluation results → {output_path}")


if __name__ == "__main__":
    main()
