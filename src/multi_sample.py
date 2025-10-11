import json, os, random
from huggingface_hub import InferenceClient

def multi_sample(n_samples=6, temperature=0.5):
    with open("data/squad_random_pairs.json") as f:
        data = json.load(f)
    token = os.getenv("HF_TOKEN")
    client = InferenceClient(model="meta-llama/Llama-3.1-8B-Instruct", token=token)
    results = []
    for item in data[:50]:
        q, gold = item["question"], item["gold"]
        gens = []
        for _ in range(n_samples):
            prompt = f"Question: {q}\nAnswer:"
            out = client.text_generation(prompt, temperature=temperature, max_new_tokens=64)
            gens.append(out.strip())
        results.append({"question": q, "gold": gold, "generations": gens})
    os.makedirs("data", exist_ok=True)
    json.dump(results, open("data/squad_multi.json", "w"), indent=2)

if __name__ == "__main__":
    multi_sample()
