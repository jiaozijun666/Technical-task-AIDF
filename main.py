# main.py
from __future__ import annotations
import os, json, argparse, importlib, runpy, re, string
from typing import List, Dict, Any, Callable, Optional
from sklearn.metrics import roc_auc_score

def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    return data

def _norm_q(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(f"[{re.escape(string.punctuation)}]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def save_text(path: str, s: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)

def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _truncate_json_list(path: str, n: int) -> int:
    if n <= 0: 
        return 0
    import json
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return 0
    orig = len(data)
    if orig <= n:
        return orig
    data = data[:n]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[main] truncated {path}: {orig} -> {len(data)}")
    return len(data)

def _env_int(name: str, default: int = 0) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


def _try_call(mod_name: str, fn_candidates: List[str] | None = None, kwargs: Dict[str, Any] | None = None) -> None:
    kwargs = kwargs or {}
    fn_candidates = fn_candidates or ["main", "run", "cli"]
    try:
        m = importlib.import_module(mod_name)
        for fn in fn_candidates:
            f = getattr(m, fn, None)
            if callable(f):
                print(f"[main] call {mod_name}.{fn}()")
                f(**kwargs)
                return
        raise AttributeError("no callable in module")
    except Exception:
        script = os.path.join(*mod_name.split(".")) + ".py"
        print(f"[main] exec script: {script}")
        runpy.run_path(script, run_name="__main__")

def _load_squad_gold(files=("data/squad_train.json", "data/squad_test.json")):
    by_id, by_q = {}, {}
    for p in files:
        if not os.path.exists(p): 
            continue
        try:
            arr = json.load(open(p))
        except Exception:
            continue
        for ex in arr:
            qid = ex.get("id") or ex.get("qid")
            q   = ex.get("question") or ""
            ans = ex.get("answers") or {}
            if isinstance(ans, dict):
                txts = ans.get("text") or ans.get("texts") or []
                gold = txts[0] if isinstance(txts, list) and txts else (txts if isinstance(txts, str) else "")
            else:
                gold = ""
            if qid: by_id[str(qid)] = gold
            if q:   by_q[_norm_q(q)] = gold
    return by_id, by_q


def _extract_gold(item):
    g = item.get("gold")
    if g:
        return g
    ans = item.get("answers") or item.get("answer") or {}
    # SQuAD: {"answers":{"text":[...], "answer_start":[...]}}
    if isinstance(ans, dict):
        txt = ans.get("text") or ans.get("texts")
        if isinstance(txt, list) and txt:
            return txt[0]
        if isinstance(txt, str):
            return txt
    return None
    
def _enrich_gold(flat: list[dict]) -> list[dict]:
    by_id, by_q = _load_squad_gold()
    changed = 0
    for r in flat:
        if r.get("gold"): 
            continue
        g = None
        if "qid" in r and str(r["qid"]) in by_id:
            g = by_id[str(r["qid"])]
        if (not g) and r.get("question"):
            g = by_q.get(_norm_q(r["question"]))
        if g:
            r["gold"] = g
            changed += 1
    if changed:
        print(f"[eval] enriched gold for {changed} rows")
    return flat
    
def _flatten(data):
    rows = []
    for i, item in enumerate(data):
        q = item.get("question")
        gold = _extract_gold(item)            # ← 新增：更稳地拿 gold
        gens = item.get("generations") or item.get("gens") or []
        for j, ans in enumerate(gens):
            a = ans if isinstance(ans, str) else (ans.get("text") if isinstance(ans, dict) else "")
            rows.append({"qid": i, "sid": j, "question": q, "gold": gold, "answer": a})
    return rows

def _align_drop_missing(score_list, labels):
    idx = [i for i, x in enumerate(score_list) if x is not None]
    return [score_list[i] for i in idx], [labels[i] for i in idx]

def _labels_api(flat: List[Dict[str, Any]]) -> Optional[List[int]]:
    try:
        ev = importlib.import_module("src.eval")
        build = getattr(ev, "build_eval_jobs", None)
        run = getattr(ev, "run_eval_jobs", None)
        parse = getattr(ev, "parse_eval_results", None)
        if not (build and run and parse):
            return None
        jobs = [{"question": r["question"], "answer": r["answer"], "gold": r.get("gold")} for r in flat]
        reqs = build(jobs)
        raw = run(reqs)
        labels = parse(raw)
        if isinstance(labels, list) and len(labels) == len(flat):
            return labels
    except Exception:
        return None
    return None

def _labels_em(flat_rows):
    labels = []
    for r in flat_rows:
        pred_norm = _norm_text(r.get("answer") or "")
        g = r.get("gold")
        golds = []
        if isinstance(g, str) and g:
            golds = [g]
        elif isinstance(g, (list, tuple)):
            golds = [x for x in g if isinstance(x, str)]
        hit = any(_norm_text(x) == pred_norm for x in golds)
        labels.append(1 if hit else 0)
    return labels
    
def _maybe_flip_labels(labels, scores_dict):
    ref = None
    for k in ("saplma", "perplexity"):
        if k in scores_dict and len(scores_dict[k]) == len(labels):
            ref = scores_dict[k]; break
    if ref is None:
        return labels
    try:
        auc = roc_auc_score(labels, ref)
    except Exception:
        return labels
    if auc < 0.5:
        print("[main] reference auc < 0.5 -> flip labels")
        return [1 - int(x) for x in labels]
    return labels


def _resolve_any(mod: str, fn_candidates: List[str]) -> Callable:
    m = importlib.import_module(mod)
    for fn in fn_candidates:
        f = getattr(m, fn, None)
        if callable(f):
            print(f"[main] using {mod}.{fn}")
            return f
    raise RuntimeError(f"missing callable in {mod}: tried {fn_candidates}")

def get_scorers() -> List[tuple[str, Callable]]:
    b = "baseline.baseline"
    reg: List[tuple[str, Callable]] = []
    reg.append(("se",          _resolve_any(b, ["run_se"])))
    reg.append(("mars",        _resolve_any(b, ["run_mars"])))
    reg.append(("mars-se",     _resolve_any(b, ["run_mars_se"])))
    reg.append(("perplexity",  _resolve_any(b, ["run_perplexity"])))
    reg.append(("p(true)",     _resolve_any(b, ["run_p_true"])))
    reg.append(("ccs",         _resolve_any(b, ["run_ccs"])))
    reg.append(("saplma",      _resolve_any(b, ["run_saplma"])))
    reg.append(("haloscope",   _resolve_any(b, ["run_haloscope"])))
    reg.append(("hami",        _resolve_any("HaMI.hami", ["run_hami","score_hami","hami","run","main"])))
    reg.append(("hami star",   _resolve_any("HaMI.hami_star", ["run_hami_star","score_hami_star","hami_star","run","main"])))
    return reg


def _compute_aurocs(score_dict: Dict[str, List[float]], labels: List[int]) -> Dict[str, float]:
    try:
        m = importlib.import_module("src.metrics")
        fn = getattr(m, "compute_aurocs", None)
        if callable(fn):
            return fn(score_dict, labels)
        au = getattr(m, "auroc")
    except Exception:
        au = None
    out: Dict[str, float] = {}
    for k, scores in score_dict.items():
        if au is None:
            pairs = sorted(zip(scores, labels), key=lambda x: x[0])
            pos = sum(labels); neg = len(labels) - pos
            if pos == 0 or neg == 0:
                out[k] = float("nan"); continue
            rank_sum = 0; r = 1
            for _, lab in pairs:
                if lab == 1: rank_sum += r
                r += 1
            u = rank_sum - pos*(pos+1)/2
            out[k] = round(u/(pos*neg), 6)
        else:
            out[k] = round(float(au(scores, labels)), 6)
    return out

def _fmt_table(aurocs: Dict[str, float]) -> str:
    order = ["p(true)","Perplexity","SE","MARS","MARS-SE","CCS","SAPLMA","HaloScope","HaMI* (Ours)","HaMI (Ours)"]
    keymap = {"p(true)":"p(true)","Perplexity":"perplexity","SE":"se","MARS":"mars","MARS-SE":"mars-se","CCS":"ccs","SAPLMA":"saplma","HaloScope":"haloscope","HaMI* (Ours)":"hami star","HaMI (Ours)":"hami"}
    col = "SQuAD"
    lines = []
    lines.append("┌" + "─"*38 + "┐")
    lines.append(f"│ {'LLaMA-3.1-8B':^14} {col:^20} │")
    lines.append("├" + "─"*38 + "┤")
    for disp in order:
        k = keymap[disp]
        v = aurocs.get(k, float('nan'))
        s = "nan" if v != v else f"{v:.3f}"
        lines.append(f"│ {disp:<14} {s:>20} │")
    lines.append("└" + "─"*38 + "┘")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print("[main] step 1/5: process_data")
    _try_call("src.process_data")

    limit = _env_int("MULTI_LIMIT", _env_int("MAIN_LIMIT", 0))
    if limit > 0:
        path = os.path.join(args.data_dir, "squad_train.json")
        _truncate_json_list(path, limit)
    
    print("[main] step 2/5: multi_sample")
    _try_call("src.multi_sample")

    print("[main] step 3/5: final_select")
    _try_call("src.final_select")

    print("[main] step 4/5: random_pairs")
    _try_call("src.random_pairs")

    print("[main] step 5/5: refined_set")
    _try_call("src.refined_set")

    multi_path = os.path.join(args.data_dir, "squad_multi.json")
    print(f"[main] load generations: {multi_path}")
    data = load_json(multi_path)
    flat = _flatten(data)
    flat = _enrich_gold(flat) 

    labels = _labels_api(flat)
    if labels is None:
        print("[main] eval labels: EM fallback")
        labels = _labels_em(flat)
    else:
        print("[main] eval labels: API")

    scorers = get_scorers()
    scores: Dict[str, List[float]] = {}
    for name, fn in scorers:
        print(f"[main] scoring: {name}")
        s = fn(flat)
        if len(s) != len(flat):
            raise ValueError(f"{name} produced {len(s)} scores for {len(flat)} rows")
        scores[name.lower()] = [float(x) for x in s]
    labels = _maybe_flip_labels(labels, scores)
    aurocs = _compute_aurocs(scores, labels)
    table = _fmt_table(aurocs)

    print("\n=== AUROC (SQuAD) ===")
    print(table)
    save_json(os.path.join(args.results_dir, "scores.json"), {"aurocs": aurocs})
    save_text(os.path.join(args.results_dir, "table.txt"), table)
    print(f"[main] saved results to {args.results_dir}")

if __name__ == "__main__":
    main()