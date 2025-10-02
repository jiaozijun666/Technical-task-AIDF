#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_select.py

Refine the 500-question sample pool (with K generated answers each) into the
final ~400-question test set, following a simple, transparent rule:

  Keep a question if at least ceil(min_valid_ratio * k) of its K samples are "valid".
  A sample is "valid" if it is non-empty and <= max_words tokens (whitespace-split).

If kept > need, downsample to exactly `need` after shuffling with a fixed seed.
If kept < need, keep all (final size < need).

Inputs:
  results/test500_samples.jsonl
    Each line: {"question": str, "gold": str, "samples": [str, ...]}

Outputs:
  data/test_final_400.jsonl
    Each line: {"question": str, "context": "", "gold": str}

Usage:
  python -m src.final_select \
    --samples results/test500_samples.jsonl \
    --out data/test_final_400.jsonl \
    --need 400 --k 6 --min_valid_ratio 0.67 --max_words 80 --seed 42
"""

import argparse
import json
import math
import os
import random
import re
from typing import List, Dict


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def word_count(s: str) -> int:
    return len(s.split())


def is_valid_answer(ans: str, max_words: int) -> bool:
    if not ans or not ans.strip():
        return False
    if word_count(ans) > max_words:
        return False
    # Optional light filters to avoid boilerplate junk; extend as needed.
    junk_patterns = [
        r"as an ai language model",
        r"i cannot|i'm unable",
        r"no context provided",
    ]
    norm = normalize(ans)
    if any(re.search(pat, norm) for pat in junk_patterns):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=str, default="results/test500_samples.jsonl",
                    help="JSONL with K samples per question.")
    ap.add_argument("--out", type=str, default="data/test_final_400.jsonl",
                    help="Output JSONL final test set.")
    ap.add_argument("--need", type=int, default=400, help="Target final size.")
    ap.add_argument("--k", type=int, default=6, help="Expected number of samples per question.")
    ap.add_argument("--min_valid_ratio", type=float, default=0.67,
                    help="Minimum fraction of valid samples required (e.g., 0.67 -> >=4/6).")
    ap.add_argument("--max_words", type=int, default=80,
                    help="Max words allowed for a sample to be considered valid.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for downsampling.")
    args = ap.parse_args()

    random.seed(args.seed)
    rows = load_jsonl(args.samples)
    keep_threshold = max(1, math.ceil(args.min_valid_ratio * args.k))

    kept = []
    total_valid_counts = []

    for ex in rows:
        q = ex.get("question", "")
        gold = ex.get("gold", "")
        samples: List[str] = ex.get("samples", []) or []

        # If file has variable K, adaptively use min(len(samples), args.k) for threshold calc
        k_here = min(len(samples), args.k) if len(samples) > 0 else args.k
        need_valid_here = max(1, math.ceil(args.min_valid_ratio * k_here))

        valid_cnt = sum(1 for s in samples[:k_here] if is_valid_answer(s, args.max_words))
        total_valid_counts.append(valid_cnt)

        if valid_cnt >= need_valid_here:
            kept.append({"question": q, "context": "", "gold": gold})

    print(f"[final_select] total={len(rows)}, kept_before_downsample={len(kept)} "
          f"(need_valid >= {keep_threshold} with k={args.k}, max_words={args.max_words})")

    # If too many, downsample to need
    if len(kept) > args.need:
        random.shuffle(kept)
        kept = kept[:args.need]

    print(f"[final_select] final_size={len(kept)} -> {args.out}")
    save_jsonl(args.out, kept)


if __name__ == "__main__":
    main()
