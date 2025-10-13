from __future__ import annotations
import os, json, time
from typing import List, Dict, Any, Optional
from prompt import build_generation_prompt
from api import chat_complete

CONFIG = {
    "INPUT_PATH": None,
    "INPUT_CANDIDATES": ["data/questions_with_ctx.json","data/squad_train.json","data/squad_test.json"],
    "OUTPUT_PATH": "data/squad_multi.json",
    "K": 5,
    "OPENAI_MODEL": "gpt-4.1",
    "TOP_P": 0.95,
    "MAX_TOKENS": 256,
    "LIMIT": 0,
    "SLEEP_BETWEEN_CALLS": 0.0,
    "MAX_RETRIES": 3,
    "BACKOFF_BASE_SEC": 1.0,
    "MODEL_DIR": os.getenv("MULTI_MODEL_DIR", ""),
    "LOG_INTERVAL": int(os.getenv("MULTI_LOG_INTERVAL", "50")),
}

def _log(s: str) -> None:
    print(f"[multi] {s}")

def _load_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    return data

def _save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _autodetect_input(cfg: dict) -> str:
    if cfg["INPUT_PATH"]:
        if os.path.isfile(cfg["INPUT_PATH"]):
            return cfg["INPUT_PATH"]
        raise FileNotFoundError(cfg["INPUT_PATH"])
    for p in cfg["INPUT_CANDIDATES"]:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("no input json")

T_CTX = "Read the context and answer concisely.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

def _build_prompt_row(row: Dict[str, Any]) -> str:
    ctx = (row.get("context") or "").strip()
    q = (row.get("question", row.get("prompt", "")) or "").strip()
    if ctx and q:
        return T_CTX.format(context=ctx, question=q)
    return build_generation_prompt(q)

def _extract_gold(row: Dict[str, Any]) -> Optional[str]:
    if "gold" in row and row["gold"] not in ("", None):
        return str(row["gold"])
    a = row.get("answers")
    if isinstance(a, dict):
        t = a.get("text")
        if isinstance(t, list) and t:
            return t[0]
    return None

_TOK = None
_MODEL = None
_DEVICE = None
_LOADED_LOCAL = False

def _ensure_local(model_dir: str):
    global _TOK, _MODEL, _DEVICE, _LOADED_LOCAL
    if _TOK is not None and _MODEL is not None:
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _log(f"loading local model: {model_dir}")
    _TOK = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if _TOK.pad_token is None:
        _TOK.pad_token = _TOK.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    _MODEL = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None, low_cpu_mem_usage=True)
    _MODEL.eval()
    _DEVICE = _MODEL.device
    _LOADED_LOCAL = True
    _log(f"local model ready on {str(_DEVICE)}")

def _gen_local(prompt: str, max_new_tokens: int, top_p: float, seed: int) -> str:
    import torch
    from transformers import set_seed
    set_seed(seed)
    inputs = _TOK(prompt, return_tensors="pt").to(_DEVICE)
    out = _MODEL.generate(
        **inputs,
        do_sample=True,
        temperature=0.5,
        top_p=float(top_p),
        max_new_tokens=int(max_new_tokens),
        eos_token_id=_TOK.eos_token_id,
        pad_token_id=_TOK.pad_token_id,
        repetition_penalty=1.0,
    )
    text = _TOK.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()

def _call_openai(prompt: str, model: str, max_tokens: int, top_p: float, max_retries: int, base_delay: float) -> str:
    for attempt in range(max_retries):
        try:
            return chat_complete(prompt=prompt, model=model, max_tokens=max_tokens, top_p=top_p)
        except Exception as e:
            msg = str(e).lower()
            if "unsupported_country_region_territory" in msg or "permissiondeniederror" in msg or "403" in msg:
                raise
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))
    return ""

def main() -> None:
    cfg = CONFIG.copy()
    in_path = _autodetect_input(cfg)
    rows = _load_json_array(in_path)
    if cfg["LIMIT"] > 0:
        rows = rows[: cfg["LIMIT"]]

    _log(f"input: {in_path}  items={len(rows)}")
    _log(f"output: {cfg['OUTPUT_PATH']}")
    _log(f"model: {cfg['OPENAI_MODEL']}  k={cfg['K']}  max_tokens={cfg['MAX_TOKENS']}  top_p={cfg['TOP_P']}")
    if cfg["MODEL_DIR"]:
        _log(f"local fallback dir: {cfg['MODEL_DIR']}")

    results: List[Dict[str, Any]] = []
    used_fallback = False

    for i, row in enumerate(rows):
        prompt = _build_prompt_row(row).strip()
        if not prompt:
            continue
        gold = _extract_gold(row)
        gens: List[str] = []
        for k in range(cfg["K"]):
            try:
                text = _call_openai(prompt=prompt, model=cfg["OPENAI_MODEL"], max_tokens=cfg["MAX_TOKENS"], top_p=cfg["TOP_P"], max_retries=cfg["MAX_RETRIES"], base_delay=cfg["BACKOFF_BASE_SEC"])
            except Exception:
                if not cfg["MODEL_DIR"]:
                    raise
                if not used_fallback:
                    _log("openai blocked → fallback to local model")
                used_fallback = True
                _ensure_local(cfg["MODEL_DIR"])
                text = _gen_local(prompt, cfg["MAX_TOKENS"], cfg["TOP_P"], seed=42 + i * cfg["K"] + k)
            gens.append(text)
            if cfg["SLEEP_BETWEEN_CALLS"] > 0:
                time.sleep(cfg["SLEEP_BETWEEN_CALLS"])
        results.append({"question": prompt, "gold": gold, "generations": gens})

        if cfg["LOG_INTERVAL"] > 0 and (i + 1) % cfg["LOG_INTERVAL"] == 0:
            _log(f"progress: {i + 1}/{len(rows)}")

    _save_json(cfg["OUTPUT_PATH"], results)
    _log(f"saved: {cfg['OUTPUT_PATH']}  items={len(results)}  k={cfg['K']}")

if __name__ == "__main__":
    main()
