# baseline/internal_representation_based/saplma.py
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.prompt import get_yes_no_prompt

def compute_saplma(model_name, data_path, output_path="results/saplma.jsonl"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    data = json.load(open(data_path))
    os.makedirs("results", exist_ok=True)
    results = []

    for item in tqdm(data, desc="Computing SAPLMA"):
        q, gold = item["question"], item["gold"]
        for g in item["generations"]:
            prompt = get_yes_no_prompt(q, gold, g)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=30)
            text = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
            score = 1.0 if "yes" in text else 0.0
            results.append({
                "question": q,
                "generation": g,
                "score": score
            })

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[✔] Saved SAPLMA results to {output_path}")
    return results
