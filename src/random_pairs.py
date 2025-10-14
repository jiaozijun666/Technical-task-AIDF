import os
import json
import random
import re

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _first_answer(item):
    # SQuAD 格式：{"answers": {"text": [..]}}
    ans = (item.get("answers") or {}).get("text") or []
    return str(ans[0]) if ans else ""

def make_random_pairs(in_path="data/squad_train.json",
                      out_path="data/squad_random_pairs.json",
                      max_pairs=5000,
                      seed=42,
                      min_len=1):
    random.seed(seed)

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for idx, it in enumerate(data):
        q = str(it.get("question") or "")
        gold = _first_answer(it)
        if len(q.strip()) < min_len or len(gold.strip()) < min_len:
            continue
        rows.append((idx, q, gold))

    n = len(rows)
    pairs = []
    seen = set()

    for i, (idx_i, q, gold) in enumerate(rows):
        for _ in range(10):  
            j = random.randrange(n)
            if j == i:
                continue
            neg = rows[j][2]  
            if _norm(neg) == _norm(gold):
                continue
            key = (idx_i, _norm(gold), _norm(neg))
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"question": q, "gold": gold, "neg": neg})
            break

        if len(pairs) >= max_pairs:
            break

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"[✔] Built {len(pairs)} random pairs → {out_path}  "
          f"(pool={n}, seed={seed})")

def main():
    in_path  = os.getenv("RPAIRS_IN",  "data/squad_train.json")
    out_path = os.getenv("RPAIRS_OUT", "data/squad_random_pairs.json")
    max_pairs = int(os.getenv("RPAIRS_MAX", "5000"))
    seed      = int(os.getenv("RPAIRS_SEED", "42"))
    min_len   = int(os.getenv("RPAIRS_MINLEN", "1"))
    make_random_pairs(in_path=in_path, out_path=out_path,
                      max_pairs=max_pairs, seed=seed, min_len=min_len)

if __name__ == "__main__":
    main()
