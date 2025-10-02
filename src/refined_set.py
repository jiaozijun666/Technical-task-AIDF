"""
refined_set.py

Build the validation set (remaining 300) from the 800 test pool:
  val_300 = test_pool_800 - test_sample_pool_500   (by exact question match)

Input :
  data/test_pool_800.jsonl
  data/test_sample_pool_500.jsonl
Output:
  data/val_300.jsonl  (size should be 300)

Run:
  python -m src.refined_set \
    --pool data/test_pool_800.jsonl \
    --sample_pool data/test_sample_pool_500.jsonl \
    --out data/val_300.jsonl
"""

import argparse
import json
import os
import random


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=str, default="data/test_pool_800.jsonl")
    ap.add_argument("--sample_pool", type=str, default="data/test_sample_pool_500.jsonl")
    ap.add_argument("--out", type=str, default="data/val_300.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--expected", type=int, default=300)
    args = ap.parse_args()

    random.seed(args.seed)

    pool = load_jsonl(args.pool)
    sample_pool = load_jsonl(args.sample_pool)

    sample_qs = {ex["question"] for ex in sample_pool}
    val = [ex for ex in pool if ex["question"] not in sample_qs]

    # Optional shuffle to make it reproducible
    random.shuffle(val)

    # Truncate/pad if needed (normally len(val) == 300)
    if len(val) > args.expected:
        val = val[:args.expected]

    save_jsonl(args.out, val)
    print(f"[refined_set] pool={len(pool)}, sample_pool={len(sample_pool)}, val={len(val)} -> {args.out}")


if __name__ == "__main__":
    main()
