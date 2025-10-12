from __future__ import annotations
import os, json, argparse, time
from typing import List, Dict, Any

# ---------------- IO ----------------
def load_questions(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        return data

def save_multi(path: str, items: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def build_prompt(q: str) -> str:
    # 如需复杂模板，可接入 src/prompt.py
    return q.strip()

# ------------- Backends -------------
# 1) Transformers (offline, local dir)
def gen_transformers_local(
    model_dir: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    repetition_penalty: float = 1.0,
    seed: int = 42,
):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # 缓存到全局，避免每次重复加载
    global _TOK, _MODEL, _DEVICE
    if "_TOK" not in globals() or _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        # 如果没有 pad/eos，做个兜底
        if _TOK.pad_token is None:
            _TOK.pad_token = _TOK.eos_token
    if "_MODEL" not in globals() or _MODEL is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        _MODEL = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        _MODEL.eval()
    if "_DEVICE" not in globals() or _DEVICE is None:
        _DEVICE = _MODEL.device

    torch.manual_seed(seed)

    inputs = _TOK(prompt, return_tensors="pt").to(_DEVICE)
    gen_out = _MODEL.generate(
        **inputs,
        do_sample=True,
        temperature=max(1e-6, float(temperature)),
        top_p=float(top_p),
        max_new_tokens=int(max_new_tokens),
        eos_token_id=_TOK.eos_token_id,
        pad_token_id=_TOK.pad_token_id,
        repetition_penalty=float(repetition_penalty)
    )
    text = _TOK.decode(gen_out[0], skip_special_tokens=True)
    # 截掉 prompt 前缀，只保留新增部分（更干净）
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()

# 2) OpenAI-compatible
def gen_openai_compat(base_url, api_key, model, prompt, temperature, top_p, max_tokens, timeout=120):
    import requests
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "n": 1,
        "stream": False,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

# 3) Ollama
def gen_ollama(model, prompt, temperature, top_p, max_tokens, base_url="http://localhost:11434", timeout=300):
    import requests
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {"temperature": temperature, "top_p": top_p, "num_predict": max_tokens},
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "").strip()

# 4) 自定义本地函数
def gen_local_func(prompt, temperature, top_p, max_tokens):
    from src.api import generate_local  # 你自己实现
    return generate_local(prompt=prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens).strip()

# ------------- Driver -------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="问题列表 JSON/JSONL（字段含 question[, gold]）")
    ap.add_argument("--out", default="data/squad_multi.json", help="输出文件（JSON）")
    ap.add_argument("--k", type=int, default=5, help="每个问题生成次数")
    ap.add_argument("--sleep", type=float, default=0.0, help="每次生成后的休眠秒数（限流）")

    ap.add_argument("--backend", choices=["transformers_local","openai_compat","ollama","local_func"],
                    default="transformers_local")

    # transformers_local
    ap.add_argument("--model_dir", default="/workspace/models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
                    help="本地模型目录（含 config.json / tokenizer.json / *.safetensors）")
    ap.add_argument("--repetition_penalty", type=float, default=1.0)

    # openai_compat
    ap.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL","http://localhost:8000"))
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY",""))
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL","gpt-3.5-turbo"))

    # ollama
    ap.add_argument("--ollama_url", default=os.environ.get("OLLAMA_URL","http://localhost:11434"))

    # decoding
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_questions(args.input)
    out_items = []

    for r in rows:
        q = str(r.get("question", r.get("prompt",""))).strip()
        if not q:
            continue
        gold = r.get("gold")
        gens = []

        for i in range(args.k):
            prompt = build_prompt(q)
            if args.backend == "transformers_local":
                text = gen_transformers_local(
                    model_dir=args.model_dir,
                    prompt=prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_tokens,
                    repetition_penalty=args.repetition_penalty,
                    seed=args.seed + i,
                )
            elif args.backend == "openai_compat":
                text = gen_openai_compat(
                    base_url=args.base_url, api_key=args.api_key, model=args.model,
                    prompt=prompt, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens
                )
            elif args.backend == "ollama":
                text = gen_ollama(
                    model=args.model, prompt=prompt, temperature=args.temperature,
                    top_p=args.top_p, max_tokens=args.max_tokens, base_url=args.ollama_url
                )
            else:  # local_func
                text = gen_local_func(prompt, args.temperature, args.top_p, args.max_tokens)

            gens.append(text)
            if args.sleep > 0:
                time.sleep(args.sleep)

        out_items.append({"question": q, "gold": gold, "generations": gens})

    save_multi(args.out, out_items)
    print(f"[save] {args.out}  (n={len(out_items)})")

if __name__ == "__main__":
    main()
