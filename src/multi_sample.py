"""
Generate K samples per question using a unified model interface.

Input JSONL (one per line, minimal schema):
  {"question": str, "context": str (optional), "gold": str (optional), "id": any (optional)}

Output JSONL (one per input line):
  {"question": str, "context": str?, "gold": str?, "id": any?, "samples": [str, ...]}

Examples
--------
python -m src.multi_sample \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --backend hf \
  --data data/interim/test_sample_pool_500.jsonl \
  --out  results/test500_samples.jsonl \
  --k 6 --temperature 0.5 --max_new_tokens 64 --with_context --limit 100

# Ollama (generation only)
python -m src.multi_sample \
  --model llama3.1:8b-instruct-q4_K_M \
  --backend ollama --api_base http://127.0.0.1:11434 \
  --data data/interim/test_sample_pool_500.jsonl \
  --out  results/test500_samples.jsonl \
  --k 6 --temperature 0.5 --max_new_tokens 64
"""

from __future__ import annotations
import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.model import get_model, GenConfig

# tqdm is optional
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x

# torch is optional (seed on HF backend)
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# --------------------------- IO helpers ---------------------------

def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def read_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
            if limit is not None and len(rows) >= limit:
                break
    return rows

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------- prompts ------------------------------

def build_prompt(question: str, context: Optional[str], with_context: bool) -> str:
    if with_context and context and str(context).strip():
        return (
            "Use the CONTEXT to answer with a short phrase ONLY.\n"
            "Do not explain. Output just the answer.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
    return (
        "Answer the question with a short phrase ONLY.\n"
        "Do not explain. Output just the answer.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


# --------------------------- core logic ---------------------------

@dataclass
class Args:
    model: str
    backend: str
    api_base: Optional[str]
    data: Path
    out: Path
    k: int
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int
    with_context: bool
    limit: Optional[int]
    seed: Optional[int]


def sample_for_row(mdl, row: Dict[str, Any], a: Args) -> Dict[str, Any]:
    q = row.get("question", "").strip()
    ctx = row.get("context", None)
    gold = row.get("gold", None)
    rid = row.get("id", None)

    prompt = build_prompt(q, ctx, a.with_context)
    cfg = GenConfig(
        temperature=a.temperature,
        top_p=a.top_p,
        top_k=a.top_k,
        max_new_tokens=a.max_new_tokens,
        do_sample=True,
        stop=None,
    )

    # Collect K generations
    outs: List[str] = []
    for _ in range(a.k):
        out = mdl.generate(prompt, cfg)
        outs.append(out.text)

    out_row: Dict[str, Any] = {"question": q, "samples": outs}
    if ctx is not None:
        out_row["context"] = ctx
    if gold is not None:
        out_row["gold"] = gold
    if rid is not None:
        out_row["id"] = rid
    return out_row


# --------------------------- CLI wiring ---------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-sample K generations per question.")
    p.add_argument("--model", required=False, default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="HF repo id or Ollama tag.")
    p.add_argument("--backend", choices=["hf", "ollama"], default="hf",
                   help="Generation backend.")
    p.add_argument("--api_base", default=None,
                   help="Ollama base URL, e.g. http://127.0.0.1:11434 (ignored for HF).")

    p.add_argument("--data", type=Path, required=True,
                   help="Input JSONL with fields: question, context? , gold? .")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSONL to write samples.")

    p.add_argument("--k", type=int, default=6, help="#samples per question.")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--with_context", action="store_true",
                   help="Include `context` field when building the prompt.")
    p.add_argument("--limit", type=int, default=None, help="Only process first N rows.")
    p.add_argument("--seed", type=int, default=None, help="Random seed (HF sampling).")
    return p


def parse_args(argv: Optional[List[str]] = None) -> Args:
    ns = build_parser().parse_args(argv)
    return Args(
        model=ns.model,
        backend=ns.backend,
        api_base=ns.api_base,
        data=ns.data,
        out=ns.out,
        k=ns.k,
        temperature=ns.temperature,
        top_p=ns.top_p,
        top_k=ns.top_k,
        max_new_tokens=ns.max_new_tokens,
        with_context=ns.with_context,
        limit=ns.limit,
        seed=ns.seed,
    )


def run(a: Args) -> None:
    # Seeding (best-effort for HF backend)
    if a.seed is not None:
        random.seed(a.seed)
        try:
            import numpy as np  # type: ignore
            np.random.seed(a.seed)
        except Exception:
            pass
        if a.backend == "hf" and torch is not None:
            try:
                torch.manual_seed(a.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(a.seed)
            except Exception:
                pass

    print(f"[multi_sample] backend={a.backend}  model={a.model}")
    print(f"[multi_sample] data={a.data}  out={a.out}")
    print(f"[multi_sample] k={a.k}  temp={a.temperature}  max_new_tokens={a.max_new_tokens}  limit={a.limit}")

    rows = read_jsonl(a.data, limit=a.limit)
    mdl = get_model(a.model, backend=a.backend, api_base=a.api_base)

    outputs: List[Dict[str, Any]] = []
    for row in tqdm(rows, desc="multi-sample", ncols=100):
        outputs.append(sample_for_row(mdl, row, a))

    write_jsonl(a.out, outputs)
    print(f"[multi_sample] wrote {len(outputs)} rows -> {a.out}")


# Entry points compatible with main.py orchestrator
def main(argv: Optional[List[str]] = None) -> int:
    a = parse_args(argv)
    run(a)
    return 0

def cli(argv: Optional[List[str]] = None) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
