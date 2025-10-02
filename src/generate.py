# -*- coding: utf-8 -*-
"""
generate.py

Minimal baseline runner with a 'perplexity' method, supporting two backends:
- hf      : Hugging Face transformers (can compute token-level loss/perplexity)
- ollama  : OpenAI-compatible API served by Ollama (generation only; no loss)

Pipeline (perplexity baseline):
  1) Build a prompt (with/without context) asking for a short phrase ONLY
  2) Generate one answer from a HF/OLLAMA model
  3) For HF backend only: compute NLL of the generated answer tokens
  4) Label = 1 if EM==0 (hallucination), else 0
  5) Report AUROC when scores are available (HF path)

Input JSONL schema per line:
  {"question": str, "context": str, "gold": str}

Outputs JSONL:
  {"question": ..., "gold": ..., "pred": ..., "em": 0/1, "loss": float | null}
"""

import os
import json
import argparse
from typing import Optional, Dict, List

# ---------- Text utils ----------

def normalize_text(s: str) -> str:
    """Very simple normalization for EM."""
    return " ".join(s.strip().lower().split())

def exact_match(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))

def build_prompt(question: str, context: Optional[str] = None, use_context: bool = False) -> str:
    """
    Build a short-answer prompt. Keeping answers short improves EM hit-rate on small models.
    """
    if use_context and context:
        return (
            "Use the CONTEXT to answer with a short phrase ONLY.\n"
            "Do not explain. Output just the answer.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
    else:
        return (
            "Answer the question with a short phrase ONLY.\n"
            "Do not explain. Output just the answer.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

# ---------- HF backend (transformers) ----------

def load_hf_model(model_name: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto" if device == "cuda" else None,
    )
    if device in ("mps", "cpu"):
        mdl.to(device)

    return tok, mdl, device

def generate_answer_hf(tokenizer, model, device: str, prompt: str, max_new_tokens: int, temperature: float) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=max(1e-6, temperature),
            pad_token_id=tokenizer.eos_token_id,
        )[0]
    gen = tokenizer.decode(out_ids[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen.strip()

def nll_of_text_hf(tokenizer, model, device: str, text: str) -> float:
    """
    Teacher-forced negative log-likelihood of `text` (HF only).
    This is a crude approximation for a short phrase.
    """
    import torch
    if not text:
        return 0.0
    ids = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**ids, labels=ids["input_ids"])
        # HF returns mean loss over tokens
        loss = float(out.loss.detach().cpu().item())
    return loss

# ---------- Ollama backend (OpenAI-compatible) ----------

def generate_answer_ollama(prompt: str, model: str, api_base: str, max_new_tokens: int, temperature: float) -> str:
    """
    Call Ollama's OpenAI-compatible /chat/completions endpoint.
    Note: no token-level loss is available from this API.
    """
    from openai import OpenAI
    client = OpenAI(base_url=api_base, api_key="ollama")  # any string works
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=float(temperature),
        max_tokens=int(max_new_tokens),
    )
    return resp.choices[0].message.content.strip()

# ---------- Runner ----------

def run_perplexity(args: argparse.Namespace) -> None:
    # Load model according to backend
    if args.backend == "hf":
        tokenizer, model, device = load_hf_model(args.model)
        print(f"[generate/perplexity] backend=hf device={device}")
    else:
        tokenizer = model = device = None
        print(f"[generate/perplexity] backend=ollama api_base={args.api_base}")

    # IO
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fin = open(args.data, "r", encoding="utf-8")
    fout = open(args.out, "w", encoding="utf-8")

    import math
    from tqdm import tqdm

    y_true: List[int] = []   # 1 if hallucination, else 0
    scores: List[float] = [] # perplexity proxy (higher means more hallu); HF only

    cnt = 0
    for line in tqdm(fin, desc="perplexity"):
        if args.limit is not None and cnt >= args.limit:
            break
        cnt += 1

        ex = json.loads(line)
        q = ex.get("question", "")
        ctx = ex.get("context", "")
        gold = ex.get("gold", "")

        prompt = build_prompt(q, ctx, use_context=args.with_context)

        # Generate
        if args.backend == "hf":
            pred = generate_answer_hf(tokenizer, model, device, prompt, args.max_new_tokens, args.temperature)
        else:
            pred = generate_answer_ollama(prompt, args.model, args.api_base, args.max_new_tokens, args.temperature)

        em = exact_match(pred, gold)

        # Score
        if args.backend == "hf":
            # Use NLL of the generated answer as a monotonic proxy of perplexity.
            loss = nll_of_text_hf(tokenizer, model, device, pred)
            scores.append(loss)       # higher loss => more hallucination
        else:
            loss = None               # Not available on Ollama

        y_true.append(1 - em)         # hallucination = 1 if EM==0

        rec = {"question": q, "gold": gold, "pred": pred, "em": em, "loss": loss}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    fin.close()
    fout.close()

    # AUROC
    if args.backend == "hf" and len(set(y_true)) > 1 and len(scores) == len(y_true):
        # Simple AUROC via sklearn if available; else, compute by ranking
        try:
            from sklearn.metrics import roc_auc_score
            auroc = float(roc_auc_score(y_true, scores))
        except Exception:
            # Fallback: compute pairwise concordance (Mann–Whitney U)
            import numpy as np
            y = np.array(y_true)
            s = np.array(scores)
            pos = s[y == 1]
            neg = s[y == 0]
            if len(pos) == 0 or len(neg) == 0:
                auroc = float("nan")
            else:
                # probability that a random positive has higher score than a random negative
                auroc = float((pos[:, None] > neg[None, :]).mean())
        print(f"[generate/perplexity] AUROC = {auroc:.4f}")
    else:
        print("[generate/perplexity] AUROC not computed (backend has no scores or single class).")

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=str, required=True, choices=["perplexity"], help="baseline to run")
    ap.add_argument("--data", type=str, required=True, help="path to input JSONL")
    ap.add_argument("--out", type=str, required=True, help="path to output JSONL")
    ap.add_argument("--model", type=str, required=True, help="model name (HF repo or Ollama tag)")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--with_context", action="store_true", help="include context in the prompt")

    # NEW: backend selection
    ap.add_argument("--backend", type=str, default="hf", choices=["hf", "ollama"],
                    help="hf=transformers; ollama=OpenAI-compatible API")
    ap.add_argument("--api_base", type=str, default="http://localhost:11434/v1",
                    help="Ollama OpenAI-compatible base url")

    args = ap.parse_args()

    if args.baseline == "perplexity":
        if args.backend == "ollama":
            print("[warn] 'perplexity' scores are unavailable on ollama backend; running generation+EM only.")
        run_perplexity(args)
    else:
        raise ValueError(f"Unknown baseline: {args.baseline}")

if __name__ == "__main__":
    main()
