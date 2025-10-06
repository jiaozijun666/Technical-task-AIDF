# baseline/uncertainty_based/p_true.py
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt import get_true_false_prompt  # ✅ 统一调用

def compute_p_true(model_name, data_path, output_path="results/p_true.jsonl"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    data = json.load(open(data_path))
    os.makedirs("results", exist_ok=True)
    true_id = tokenizer(" True", add_special_tokens=False).input_ids[0]
    false_id = tokenizer(" False", add_special_tokens=False).input_ids[0]

    results = []
    for item in tqdm(data, desc="Computing p(True)"):
        q = item["question"]
        for g in item["generations"]:
            prompt = get_true_false_prompt(q, g)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits[:, -1, :].softmax(dim=-1).squeeze()
            p_true = logits[true_id].item()
            p_false = logits[false_id].item()
            score = p_true / (p_true + p_false + 1e-8)
            results.append({
                "question": q,
                "generation": g,
                "score": float(score)
            })

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[✔] Saved p(True) results to {output_path}")
    return results
