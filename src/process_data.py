from datasets import load_dataset
import json, os

def load_and_sample_squad(n_train=2000, n_test=800, seed=42):
    dataset = load_dataset("rajpurkar/squad")
    train = dataset["train"].shuffle(seed=seed).select(range(n_train))
    test = dataset["validation"].shuffle(seed=seed).select(range(n_test))
    os.makedirs("data", exist_ok=True)
    json.dump(train.to_list(), open("data/squad_train.json", "w"), indent=2)
    json.dump(test.to_list(), open("data/squad_test.json", "w"), indent=2)

def main():
    n_train = int(os.getenv("DATA_TRAIN_N", "2000"))
    n_test  = int(os.getenv("DATA_TEST_N",  "800"))
    seed    = int(os.getenv("DATA_SEED",   "42"))
    load_and_sample_squad(n_train=n_train, n_test=n_test, seed=seed)

if __name__ == "__main__":
    main()

