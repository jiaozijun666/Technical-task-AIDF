import os
import json
from tqdm import tqdm
from src.model import get_model, GenConfig
from src.prompt import build_eval_prompt

def generate_multiple_answers(data, model_id="meta-llama/Llama-3.2-1B-Instruct", n=6, output_path="data/squad_multi_debug.json"):
    """
    Generate multiple diverse answers per question.
    Each question will have `n` sampled generations.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model = get_model(model_id)
    print(f"[INFO] Loaded model: {model_id}")

    results = []
    for item in tqdm(data, desc="Generating multi-sample answers"):
        q = item["question"]
        gold = item.get("gold", "")

        generations = []
        for _ in range(n):
            prompt = build_eval_prompt(q)
            cfg = GenConfig(temperature=0.7, top_p=0.9, max_new_tokens=64)
            out = model.generate(prompt, cfg)
            generations.append(out.text.strip())

        results.append({
            "question": q,
            "gold": gold,
            "generations": generations
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved multi-sample data to {output_path}")
    print(f"[INFO] Total questions processed: {len(results)}")


if __name__ == "__main__":
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad")["validation"]
    data = [{"question": d["question"], "gold": d["answers"]["text"][0]} for d in ds.select(range(50))]  # subset

    generate_multiple_answers(data)
