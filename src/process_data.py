# src/process_data.py

import os
import json
import random
from typing import List, Dict, Tuple
from datasets import load_dataset


def load_squad():
    """Load the SQuAD dataset from HuggingFace."""
    return load_dataset("rajpurkar/squad")


def preprocess_split(ds_split, n_samples: int, seed: int = 42) -> List[Dict]:
    """Sample n examples from a split and convert to (question, context, gold)."""
    idx = list(range(len(ds_split)))
    random.Random(seed).shuffle(idx)
    idx = idx[:n_samples]

    data = []
    for i in idx:
        ex = ds_split[i]
        q = ex["question"]
        ctx = ex["context"]
        gold = ex["answers"]["text"]  # list of possible answers
        data.append({"question": q, "context": ctx, "gold": gold})
    return data


def preprocess_squad(n_train: int = 200, n_val: int = 80, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Load SQuAD and preprocess train + validation subsets."""
    ds = load_squad()
    train = preprocess_split(ds["train"], n_train, seed=seed)
    val = preprocess_split(ds["validation"], n_val, seed=seed + 1)
    return train, val


def save_jsonl(path: str, items: List[Dict]) -> None:
    """Save a list of dicts to JSONL format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


if __name__ == "__main__":
    # Quick test
    train, val = preprocess_squad(n_train=50, n_val=20, seed=42)
    print(f"train={len(train)}, val={len(val)}")
    print("example:", train[0]["question"])

    # Save to disk
    save_jsonl("data/squad_train_50.jsonl", train)
    save_jsonl("data/squad_val_20.jsonl", val)
    print("Saved to data/ ...")

    # Reload for sanity check
    loaded = load_jsonl("data/squad_train_50.jsonl")
    print("Reloaded example:", loaded[0])
