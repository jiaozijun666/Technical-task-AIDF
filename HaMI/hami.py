import os
from tqdm import tqdm
from src.model import get_model, GenConfig
from src.prompt import build_hami_prompt


def compute_hami(pairs, model_id="meta-llama/Llama-3.2-1B-Instruct", save_path="results/hami_results.json"):
    """
    HaMI baseline: Compare two candidate answers (A and B) for each question,
    and let the model choose which is more correct.
    """

    model = get_model(model_id, backend="hf")
    results = []

    for item in tqdm(pairs, desc="HaMI baseline"):
        q = item.get("question", "")
        gens = item.get("generations", [])
        if len(gens) < 2:
            continue

        # Pairwise comparison for each consecutive generation
        for i in range(len(gens) - 1):
            a_true, a_false = gens[i], gens[i + 1]
            prompt = build_hami_prompt(q, a_true, a_false)

            cfg = GenConfig(temperature=0.3, top_p=1.0, max_new_tokens=10)
            out = model.generate(prompt, cfg).text.strip()

            pred = "A" if "A" in out else ("B" if "B" in out else "unknown")
            results.append({
                "question": q,
                "A": a_true,
                "B": a_false,
                "pred": pred,
                "raw_output": out
            })

    # save results
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    import json
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved HaMI results → {save_path}")
    return results


if __name__ == "__main__":
    import json
    input_path = "data/squad_final.json"
    pairs = json.load(open(input_path))
    compute_hami(pairs)
