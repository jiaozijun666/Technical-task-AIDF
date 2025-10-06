# baseline/internal_representation_based/ccs.py
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def compute_ccs(model_name, data_path, output_path="results/ccs.jsonl"):
    """
    Contrastive Confidence Scoring (CCS)
    Measures the difference between the logit for the generated token
    and the overall log-probability mass, as a proxy for confidence.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    with open(data_path, "r") as f:
        data = json.load(f)

    results = []
    os.makedirs("results", exist_ok=True)

    for item in tqdm(data, desc="Computing CCS"):
        q = item["question"]
        for g in item["generations"]:
            prompt = f"Answer the question: {q}"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            labels = tokenizer(g, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[:, -labels.shape[1]:, :]

                # token-level positive log-prob
                pos = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                pos_logp = pos.mean().item()

                # full softmax log-sum-exp
                neg_logp = torch.logsumexp(logits, dim=-1).mean().item()

                score = pos_logp - neg_logp  # margin
                results.append({
                    "question": q,
                    "generation": g,
                    "score": float(score)
                })

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[✔] Saved CCS results to {output_path}")
    return results
