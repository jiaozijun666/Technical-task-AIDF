import os, json, re
from tqdm import tqdm

def _as_text(x):
    if isinstance(x, dict) and "text" in x:
        return str(x["text"])
    return str(x)

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def refine_data(in_path="data/squad_multi.json", out_path="data/squad_refined.json", min_len=6):
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    refined = []
    for item in tqdm(data, desc="Refining"):
        gens_raw = item.get("generations") or []
        gens_txt = [_as_text(g).strip() for g in gens_raw if _as_text(g) and len(_as_text(g).strip()) >= min_len]

        seen = set()
        gens = []
        for t in gens_txt:
            key = _norm(t)
            if key and key not in seen:
                seen.add(key)
                gens.append(t)

        if len(gens) >= 2:
            refined.append({
                "qid": int(item.get("qid") or item.get("id") or len(refined)),
                "question": item.get("question") or "",
                "gold": (item.get("gold") or (item.get("answers") or [""]))[0]
                        if isinstance(item.get("answers"), list) else (item.get("gold") or ""),
                "generations": gens
            })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(refined, f, ensure_ascii=False, indent=2)
    print(f"[✔] Saved {len(refined)} refined samples to {out_path}")

def main():
    in_path  = os.getenv("REFINE_IN",  "data/squad_multi.json")
    out_path = os.getenv("REFINE_OUT", "data/squad_refined.json")
    min_len  = int(os.getenv("REFINE_MIN_LEN", "6"))
    refine_data(in_path=in_path, out_path=out_path, min_len=min_len)

if __name__ == "__main__":
    main()
