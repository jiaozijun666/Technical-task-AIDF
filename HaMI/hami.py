import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.prompt import get_hami_instruction

def train_and_evaluate_hami(data_path="data/squad_refined.json",
                            model_name="meta-llama/Llama-3.1-8B-Instruct",
                            output_path="results/hami.jsonl"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    data = json.load(open(data_path))
    results = []
    os.makedirs("results", exist_ok=True)

    for item in tqdm(data, desc="Running HaMI Evaluation"):
        q, gen, label = item["question"], item["generation"], item["label"]
        prompt = get_hami_instruction(q, gen, label)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
            mean_logit = logits.mean().item()
        results.append({
            "question": q,
            "generation": gen,
            "label": label,
            "score": float(mean_logit)
        })

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[✔] Saved HaMI evaluation results to {output_path}")
    return results
