import json, random, os

def make_random_pairs():
    with open("data/squad_train.json") as f:
        data = json.load(f)
    pairs = []
    for item in data:
        q = item["question"]
        gold = item["answers"]["text"][0]
        wrong = random.choice(data)["answers"]["text"][0]
        if wrong == gold: continue
        pairs.append({"question": q, "gold": gold, "neg": wrong})
    os.makedirs("data", exist_ok=True)
    json.dump(pairs, open("data/squad_random_pairs.json", "w"), indent=2)

if __name__ == "__main__":
    make_random_pairs()
