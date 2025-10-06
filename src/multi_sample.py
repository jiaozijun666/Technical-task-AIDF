# src/multi_sample.py

import os
import json
from tqdm import tqdm
from typing import List
from src.model import get_model, GenConfig
from src.prompt import get_generation_prompt


def load_data(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run process_data.py first.")
    with open(path, "r") as f:
        return json.load(f)


def generate_multiple_answers(
    data: List[dict],
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    n_samples: int = 6,
    output_path: str = "data/squad_multi.json",
    backend: str = "hf",
):
    """
    Generate multiple answers for each question using a unified model interface.
    Works with both Hugging Face and Ollama backends.
    """
    print(f"Loading model via unified client: {model_id} (backend={backend})")
    model = get_model(model_id, backend=backend)
    cfg = GenConfig(max_new_tokens=64, temperature=0.6, top_p=0.9, do_sample=True)

    results = []
    for item in tqdm(data, desc="Generating multiple samples"):
        question = item["question"]
        gold = item["answers"]["text"][0] if "answers" in item else item["gold"]
        generations = []

        for _ in range(n_samples):
            prompt = get_generation_prompt(question)
            out = model.generate(prompt, cfg)
            generations.append(out.text)

        results.append({
            "question": question,
            "gold": gold,
            "generations": generations
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[✔] Saved {len(results)} questions with multi-samples to {output_path}")


if __name__ == "__main__":
    input_path = "data/squad_test.json"
    data = load_data(input_path)
    generate_multiple_answers(data)
