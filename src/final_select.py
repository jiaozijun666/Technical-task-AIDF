import os
import json
import random

def load_multi_sample(path="data/squad_multi.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run multi_sample.py first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _as_text(x):
    if isinstance(x, dict) and "text" in x:
        return str(x["text"])
    return str(x)

def filter_generations(generations, min_len=10):
    filtered = []
    for g in generations:
        text = _as_text(g).strip()
        if len(text) < min_len:
            continue
        tl = text.lower()
        if "sorry" in tl or "as an ai" in tl or "language model" in tl:
            continue
        filtered.append(text)
    return filtered

def select_high_quality(data, sample_size=500, retain_size=400, seed=42, output_path="data/quadru_pairs.json", min_len=10):
    random.seed(seed)
    sampled = random.sample(data, min(sample_size, len(data)))
    final_data = []
    for item in sampled:
        gens_raw = item.get("generations") or []
        gens = filter_generations(gens_raw, min_len=min_len)
        if not gens:
            continue
        final_data.append({
            "question": item.get("question") or "",
            "gold": (item.get("gold") or (item.get("answers") or [""]))[0] if isinstance(item.get("answers"), list) else (item.get("gold") or ""),
            "generations": gens
        })
    if len(final_data) > retain_size:
        final_data = random.sample(final_data, retain_size)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    print(f"[✔] Selected {len(final_data)} clean samples saved to {output_path}")

def main():
    in_path = os.getenv("FINAL_IN", "data/squad_multi.json")
    out_path = os.getenv("FINAL_OUT", "data/quadru_pairs.json")
    sample_size = int(os.getenv("FINAL_SAMPLE_SIZE", "500"))
    retain_size = int(os.getenv("FINAL_RETAIN_SIZE", "400"))
    seed = int(os.getenv("FINAL_SEED", "42"))
    min_len = int(os.getenv("FINAL_MIN_LEN", "10"))
    data = load_multi_sample(in_path)
    select_high_quality(data, sample_size=sample_size, retain_size=retain_size, seed=seed, output_path=out_path, min_len=min_len)

if __name__ == "__main__":
    main()
