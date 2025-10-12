# src/multi_sample.py
# Offline multi-sampling with a local Transformers model directory (no HF Hub).
# - Fixed temperature = 0.5 (paper setting)
# - Input: JSON or JSONL [{ "question"[, "gold"] }, ...]
# - Output: JSON [{ "question", "gold", "generations": [str, ...] }, ...]

from __future__ import annotations
import os
import json
import time
import argparse
from typing import List, Dict, Any

FIXED_TEMPERATURE = 0.5  # paper setting

# ---------------- I/O ----------------
def load_questions(path: str) -> List[Dict[str, Any]]:
    """Load a list of items containing 'question' (and optional 'gold')."""
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "Input must be a list of objects"
    return data

def save_multi(path: str, items: List[Dict[str, Any]]) -> None:
    """Save multi-sample results as a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def build_prompt(question: str) -> str:
    """Return the prompt for generation; adjust here if you need a template."""
    return question.strip()

# -------- Local Transformers backend (offline) --------
_TOK = None
_MODEL = None
_DEVICE = None

def _ensure_model(model_dir: str) -> None:
    """Load tokenizer/model from a local directory once."""
    global _TOK, _MODEL, _DEVICE
    if _TOK is not None and _MODEL is not None:
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    _TOK = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if _TOK.pad_token is None:
        _TOK.pad_token = _TOK.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    _MODEL.eval()
    _DEVICE = _MODEL.device

def generate_local(
    model_dir: str,
    prompt: str,
    top_p: float,
    max_new_tokens: int,
    repetition_penalty: float,
    seed: int,
) -> str:
    """Generate one completion from a local Transformers model directory."""
    import torch
    from transformers import set_seed

    _ensure_model(model_dir)
    set_seed(seed)

    inputs = _TOK(prompt, return_tensors="pt").to(_DEVICE)
    out = _MODEL.generate(
        **inputs,
        do_sample=True,
        temperature=FIXED_TEMPERATURE,   # fixed at 0.5
        top_p=float(top_p),
        max_new_tokens=int(max_new_tokens),
        eos_token_id=_TOK.eos_token_id,
        pad_token_id=_TOK.pad_token_id,
        repetition_penalty=float(repetition_penalty),
    )
    text = _TOK.decode(out[0], skip_special_tokens=True)
    # Keep only the newly generated continuation for cleanliness:
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()

# ---------------- Driver ----------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSON/JSONL with items containing 'question'[, 'gold']")
    ap.add_argument("--out", default="data/squad_multi.json", help="Output JSON path")
    ap.add_argument("--k", type=int, default=5, help="Number of samples per question")
    ap.add_argument("--model_dir", required=True, help="Local model dir (config.json/tokenizer.json/*.safetensors)")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between calls (rate limiting)")
    args = ap.parse_args()

    rows = load_questions(args.input)
    results = []

    for idx, row in enumerate(rows):
        q = str(row.get("question", row.get("prompt", ""))).strip()
        if not q:
            continue
        gold = row.get("gold")

        generations: List[str] = []
        for i in range(args.k):
            prompt = build_prompt(q)
            text = generate_local(
                model_dir=args.model_dir,
                prompt=prompt,
                top_p=args.top_p,
                max_new_tokens=args.max_tokens,
                repetition_penalty=args.repetition_penalty,
                seed=args.seed + i,
            )
            generations.append(text)
            if args.sleep > 0:
                time.sleep(args.sleep)

        results.append({"question": q, "gold": gold, "generations": generations})

    save_multi(args.out, results)
    print(f"[save] {args.out}  (questions={len(results)}, k={args.k}, temperature={FIXED_TEMPERATURE})")

if __name__ == "__main__":
    main()
