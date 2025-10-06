import os
import json
from tqdm import tqdm
from typing import List
from src.model import get_model, GenConfig
from src.prompt import get_generation_prompt


def load_data(path: str) -> List[dict]:
    """Load SQuAD-style dataset from JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run process_data.py first.")
    with open(path, "r") as f:
        return json.load(f)


def generate_multiple_answers(data: List[dict]):
    """
    Lightweight version for Colab: uses a small model, and only a few samples.
    """
    model_id = "meta-llama/Llama-3.2-1B-Instruct"   
    n_samples = 3                                 
    max_questions = 5                               
    output_path = "data/squad_multi_debug.json"
    backend = "hf"

    print(f"[Light Test Mode] Using model: {model_id}")
    print(f"Generating {n_samples} answers for {max_questions} questions...\n")

    model = get_model(model_id, backend=backend)
    cfg = GenConfig(max_new_tokens=50, temperature=0.7, top_p=0.9, do_sample=True)

    results = []
    for item in tqdm(data[:max_questions], desc="Generating samples"):
        question = item["question"]
        gold = item["answers"]["text"][0] if "answers" in item else item.get("gold", "")
        generations = []

        for i in range(n_samples):
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

    print(f"Saved {len(results)} questions with multiple generations to {output_path}")


if __name__ == "__main__":
    input_path = "data/squad_test.json"
    data = load_data(input_path)
    generate_multiple_answers(data)
