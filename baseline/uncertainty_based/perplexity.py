import os
import torch
import math
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def compute_perplexity(model_name, data_path, output_path="results/perplexity.jsonl"):
    """
    Compute perplexity-based hallucination detection scores.
    Lower PPL => more confident => likely factual.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    data = json.load(open(data_path))
    results = []

    for item in tqdm(data, desc="Computing Perplexity"):
        q = item["question"]
        for g in item["generations"]:
            text = f"Question: {q}\nAnswer: {g}"
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                loss = model(**inputs, labels=inputs["input_ids"]).loss
            ppl = math.exp(loss.item())
            results.append({
                "question": q,
                "generation": g,
                "score": ppl
            })

    os.makedirs("results", exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[✔] Saved perplexity scores to {output_path}")
    return results
