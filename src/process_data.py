# src/process_data.py
from datasets import load_dataset
import json, os

def load_and_sample_squad(n_train=2000, n_test=800, seed=42):
    dataset = load_dataset("rajpurkar/squad")
    train = dataset["train"].shuffle(seed=seed).select(range(n_train))
    test = dataset["validation"].shuffle(seed=seed).select(range(n_test))
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    print("Example sample:")
    print("Q:", train[0]["question"])
    print("A:", train[0]["answers"]["text"][0])
    return train, test

if __name__ == "__main__":
    train, test = load_and_sample_squad()

    os.makedirs("data", exist_ok=True)

    # ✅ 转为 list[dict] 再保存
    with open("data/squad_train.json", "w") as f:
        json.dump(train.to_list(), f, indent=2)
    with open("data/squad_test.json", "w") as f:
        json.dump(test.to_list(), f, indent=2)

    print("[✔] Saved data/squad_train.json and data/squad_test.json")
