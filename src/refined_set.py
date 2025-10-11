import json, os
from tqdm import tqdm

def refine_data():
    with open("data/squad_multi.json") as f:
        data = json.load(f)
    refined = []
    for item in tqdm(data, desc="Refining"):
        gens = list({g for g in item["generations"] if len(g.strip()) > 5})
        if len(gens) >= 2:
            refined.append({"question": item["question"], "gold": item["gold"], "generations": gens})
    json.dump(refined, open("data/squad_refined.json", "w"), indent=2)
    print(f"Saved {len(refined)} refined samples.")

if __name__ == "__main__":
    refine_data()
