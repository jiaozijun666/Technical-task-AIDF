import os
import json
from tqdm import tqdm

def refine_data(input_path: str, output_path: str):
    """
    Refine the multi-sample dataset:
    - Removes empty or duplicate generations
    - Filters low-quality entries
    - Saves refined results to output_path
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found. Run multi_sample.py first.")

    with open(input_path, "r") as f:
        data = json.load(f)

    refined = []
    for item in tqdm(data, desc="Refining dataset"):
        q = item.get("question", "")
        gold = item.get("gold", "")
        gens = item.get("generations", [])

        
        gens = list({g.strip() for g in gens if isinstance(g, str) and len(g.strip()) > 5})

        
        if len(gens) < 2:
            continue

        refined.append({
            "question": q,
            "gold": gold,
            "generations": gens
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(refined, f, indent=2, ensure_ascii=False)

    print(f"Saved refined dataset with {len(refined)} examples → {output_path}")


if __name__ == "__main__":
    input_path = "data/squad_multi_debug.json"  
    output_path = "data/squad_final.json"      
    refine_data(input_path, output_path)
