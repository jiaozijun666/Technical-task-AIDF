#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — Batch runner at project root.

Runs multiple baselines by calling src.generate.run_eval(...) directly.

Example (HF backend):
  python main.py \
    --data data/val_300.jsonl \
    --out_dir results \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --backend hf \
    --baselines perplexity p_true mars mars_se \
    --with_context --temperature 0.5 --max_new_tokens 16 --limit 100

Example (Ollama backend; only methods not needing token-level loss):
  python main.py \
    --data data/val_300.jsonl \
    --out_dir results \
    --model llama3.1:8b-instruct-q4_K_M \
    --backend ollama \
    --baselines p_true \
    --with_context --limit 100
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict

# --- Make sure project root is on sys.path, so 'src' is importable ---
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generate import run_eval  # noqa: E402  (import after path fix)

SUPPORTED = {"perplexity", "p_true", "mars", "mars_se"}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multiple baselines and aggregate AUROC.")
    p.add_argument("--data", required=True, help="Input JSONL with keys: question, context?, gold?")
    p.add_argument("--out_dir", required=True, help="Directory to store outputs.")
    p.add_argument("--model", required=True, help="HF model id or Ollama model tag.")
    p.add_argument("--backend", choices=["hf", "ollama"], default="hf", help="Generation backend.")
    # Accept both space-separated and comma-separated baseline lists
    p.add_argument("--baselines", nargs="+", required=True,
                   help="Baselines to run (space-separated). Supported: perplexity p_true mars mars_se")
    p.add_argument("--with_context", action="store_true", help="Include 'context' in the prompt if present.")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--api_base", default="http://127.0.0.1:11434",
                   help="Ollama base URL (when backend=ollama).")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    # Normalize baseline list (flatten possible comma-separated tokens)
    requested: List[str] = []
    for token in args.baselines:
        requested.extend([t.strip() for t in token.split(",") if t.strip()])
    requested = [b.lower() for b in requested]

    # Validate
    unknown = [b for b in requested if b not in SUPPORTED]
    if unknown:
        raise ValueError(f"Unknown baseline(s): {unknown}. Supported: {sorted(SUPPORTED)}")

    # Filter HF-only methods when backend is Ollama
    if args.backend == "ollama":
        hf_only = {"perplexity", "mars", "mars_se"}
        requested = [b for b in requested if b not in hf_only]
        if not requested:
            print("[main] No baselines left to run on Ollama "
                  "(perplexity/mars/mars_se require token-level scores via HF).")
            return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.data).stem  # e.g., val_300

    print(f"[main] data={args.data}")
    print(f"[main] out_dir={out_dir}")
    print(f"[main] backend/model={args.backend}/{args.model}")
    print(f"[main] baselines={requested}")

    summary: Dict[str, float] = {}
    for b in requested:
        out_path = out_dir / f"{stem}_{b}.jsonl"
        print(f"\n=== Running: {b} -> {out_path} ===")
        try:
            auroc = run_eval(
                baseline=b,
                data=args.data,
                out=str(out_path),
                model=args.model,
                backend=args.backend,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                with_context=args.with_context,
                limit=args.limit,
                api_base=args.api_base,
            )
            summary[b] = auroc
        except Exception as e:
            print(f"[main] baseline '{b}' failed: {e}")
            summary[b] = float("nan")

    print("\n===== SUMMARY (AUROC) =====")
    for b in requested:
        a = summary.get(b, float("nan"))
        print(f"{b:12s} : {a:.4f}" if a == a else f"{b:12s} : NaN")
    print("===========================\n")

if __name__ == "__main__":
    main()
