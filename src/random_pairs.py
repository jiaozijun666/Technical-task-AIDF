"""
random_pairs.py

Create train / test pools following the paper's protocol (SQuAD only):
  - train: 2000 QA pairs
  - test_pool: 800 QA pairs
  - from test_pool -> sample_pool: 500 QA (for multi-sampling, k=6 later)
  - remaining 300 QA -> validation set

Outputs (JSONL, one object per line: {"question","context","gold"}):
  data/train_2000.jsonl
  data/test_pool_800.jsonl
  data/test_sample_pool_500.jsonl
  data/val_300.jsonl

Run:
  python -m src.random_pairs --seed 42 --out_dir data
"""

import argparse
import json
import os
import random
from typing import List, Dict

from datasets import load_dataset


def _to_items(squad_split) -> List[Dict]:
    """
    Convert HF SQuAD split to our minimal schema.
    We keep context in case you want to inspect, but the prompts will be zero-shot later.
    """
    items: List[Dict] = []
    for ex in squad_split:
        q = (ex.get("question") or "").strip()
        ctx = (ex.get("context") or "").strip()
        answers = (ex.get("answers") or {}).get("text", [])
        if not q or not answers:
            continue
        gold = (answers[0] or "").strip()
        if not gold:
            continue
        items.append({"question": q, "context": ctx, "gold": gold})
    return items


def _dedup_by_question(items: List[Dict]) -> List[Dict]:
    """Remove duplicates by exact question text (stable order)."""
    seen = set()
    uniq: List[Dict] = []
    for ex in items:
        key = ex["question"]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(ex)
    return uniq


def _save_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    ap.add_argument("--out_dir", type=str, default="data", help="Directory to write JSONL files.")
    ap.add_argument("--train_n", type=int, default=2000, help="Number of train QA pairs.")
    ap.add_argument("--test_pool_n", type=int, default=800, help="Number of test-pool QA pairs.")
    ap.add_argument("--sample_pool_n", type=int, default=500, help="Subset of test_pool used for multi-sampling.")
    ap.add_argument("--val_n", type=int, default=300, help="Validation size (remaining of test_pool).")
    ap.add_argument("--dataset", type=str, default="rajpurkar/squad", help="HF dataset path.")
    args = ap.parse_args()

    # Basic sanity
    if args.sample_pool_n + args.val_n != args.test_pool_n:
        raise ValueError(
            f"Require sample_pool_n + val_n == test_pool_n "
            f"({args.sample_pool_n} + {args.val_n} != {args.test_pool_n})"
        )

    random.seed(args.seed)

    # Load SQuAD
    ds = load_dataset(args.dataset)
    # Combine train + validation to increase pool for random sampling
    items = _to_items(ds["train"]) + _to_items(ds["validation"])
    items = _dedup_by_question(items)

    if len(items) < args.train_n + args.test_pool_n:
        raise RuntimeError(
            f"Not enough items ({len(items)}) to sample "
            f"{args.train_n}+{args.test_pool_n}."
        )

    # Shuffle once for all downstream draws
    random.shuffle(items)

    # Sample splits
    train = items[: args.train_n]
    pool = items[args.train_n : args.train_n + args.test_pool_n]
    sample_pool = pool[: args.sample_pool_n]  # for multi-sampling (k=6 later)
    val = pool[args.sample_pool_n : args.sample_pool_n + args.val_n]  # remaining 300

    # Persist
    _save_jsonl(os.path.join(args.out_dir, "train_2000.jsonl"), train)
    _save_jsonl(os.path.join(args.out_dir, "test_pool_800.jsonl"), pool)
    _save_jsonl(os.path.join(args.out_dir, "test_sample_pool_500.jsonl"), sample_pool)
    _save_jsonl(os.path.join(args.out_dir, "val_300.jsonl"), val)

    # Manifest (optional)
    manifest = {
        "seed": args.seed,
        "train_n": len(train),
        "test_pool_n": len(pool),
        "sample_pool_n": len(sample_pool),
        "val_n": len(val),
        "dataset": args.dataset,
        "note": "Use zero-shot prompts and temperature=0.5 for generation later.",
    }
    _save_jsonl(os.path.join(args.out_dir, "manifest_random_pairs.jsonl"), [manifest])

    print(
        f"Done. train={len(train)}, test_pool={len(pool)}, "
        f"sample_pool={len(sample_pool)}, val={len(val)} -> saved under {args.out_dir}/"
    )


if __name__ == "__main__":
    main()
