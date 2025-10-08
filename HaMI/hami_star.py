import os
from tqdm import tqdm
from src.model import get_model, GenConfig
from src.prompt import build_hami_star_prompt


def compute_hami_star(pairs, model_id="meta-llama/Llama-3.2-1B-Instruct", save_path="results/hami_star_results.json"):
    """
    HaMI* baseline: Enhanced version using reasoning-based prompt;
    Model decides whether generation 1 or 2 is more factual.
    """

    model = get_model(model_id, backend="hf")
    results = []

    for item in tqdm(pairs, desc="HaMI* baseline"):
        q = item.get("question", "")
        gens = item.get("generations", [])
        if len(gens) < 2:
            continue

        for i in range(len(gens) - 1):
            a1, a2 = gens[i], gens[i + 1]
            prompt = build_hami_star_prompt(q, a1, a2)

            cfg = GenConfig(temperature=0.3, top_p=1.0, max_new_tokens=10)
            out = model.generate(prompt, cfg).text.strip()

            pred = "1" if "1" in out else ("2" if "2" in out else "unknown")
            results.append({
                "question": q,
                "ans_1": a1,
                "ans_2": a2,
                "pred": pred,
                "raw_output": out
            })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    import json
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved HaMI* results → {save_path}")
    return results


if __name__ == "__main__":
    import json
    input_path = "data/squad_final.json"
    pairs = json.load(open(input_path))
    compute_hami_star(pairs)
