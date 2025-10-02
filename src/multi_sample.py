"""
multi_sample.py

For each question in the 500-size sample pool, generate K answers:
  - zero-shot (context-free) prompt as in the paper
  - temperature = 0.5
  - do_sample = True

Input : data/test_sample_pool_500.jsonl  (one JSON per line: {question, context, gold})
Output: results/test500_samples.jsonl     (one JSON per line: {question, gold, samples: [str,...]})

Run:
  python -m src.multi_sample \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --data data/test_sample_pool_500.jsonl \
    --out results/test500_samples.jsonl \
    --k 6 --temperature 0.5 --max_new_tokens 64
"""

import argparse
import json
import os
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **_: x  # fallback no-op


def build_prompt_qa_no_context(question: str) -> str:
    """Zero-shot prompt (no context), following the paper."""
    return (
        "Answer the following question in a single but complete sentence.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def sample_k(
    tokenizer,
    model,
    prompt: str,
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float = 1.0,
) -> List[str]:
    """Generate K samples for a single prompt."""
    dev = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(dev)
    in_len = enc["input_ids"].shape[1]
    outs = []
    for _ in range(k):
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()
        outs.append(text)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="HF model id, e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--data", type=str, default="data/test_sample_pool_500.jsonl")
    ap.add_argument("--out", type=str, default="results/test500_samples.jsonl")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()

    device = pick_device()
    print(f"[multi_sample] device={device}")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(args.model)
    mdl.to(device).eval()

    rows = load_jsonl(args.data)
    out_rows = []

    for ex in tqdm(rows, desc="Sampling"):
        q = ex["question"]
        prompt = build_prompt_qa_no_context(q)
        samples = sample_k(
            tok, mdl, prompt,
            k=args.k,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        out_rows.append({"question": q, "gold": ex["gold"], "samples": samples})

    save_jsonl(args.out, out_rows)
    print(f"[multi_sample] saved {len(out_rows)} items to {args.out}")


if __name__ == "__main__":
    main()
