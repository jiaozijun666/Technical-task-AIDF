from datasets import load_dataset
import json, os

def load_and_sample_squad(n_train=2000, n_test=800, seed=42):
    dataset = load_dataset("rajpurkar/squad")
    train = dataset["train"].shuffle(seed=seed).select(range(n_train))
    test = dataset["validation"].shuffle(seed=seed).select(range(n_test))
    os.makedirs("data", exist_ok=True)
    json.dump(train.to_list(), open("data/squad_train.json", "w"), indent=2)
    json.dump(test.to_list(), open("data/squad_test.json", "w"), indent=2)

if __name__ == "__main__":
    load_and_sample_squad()
