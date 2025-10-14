import os, argparse, json, math
from typing import List, Dict, Any
from importlib import import_module

from src.data import load_json, save_json, save_text
from src.utils import exact_match
from src.metrics import score_p_t, score_p_s, score_p_c, compute_aurocs_with_labels

from baseline.baseline import (
    run_se, run_mars, run_mars_se, run_perplexity, run_p_true,
    run_ccs, run_saplma, run_haloscope, _get_local_model
)
from HaMI.hami import run_hami
from HaMI.hami_star import run_hami_star



def _env_int(k: str, default: int) -> int:
    v = os.getenv(k)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


def _truncate_json_list(path: str, limit: int) -> None:
    if not limit or limit <= 0:
        return
    if not os.path.exists(path):
        return
    data = json.load(open(path, "r"))
    if isinstance(data, list) and len(data) > limit:
        json.dump(data[:limit], open(path, "w"))


def _flatten(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for q in data:
        qid = int(q.get("qid") or q.get("id") or len(out))
        ctx = q.get("context") or q.get("ctx") or ""
        question = q.get("question") or ""
        gens = q.get("generations") or []
        golds = q.get("answers") or q.get("golds") or []
        for g in gens:
            out.append({
                "qid": qid,
                "question": question,
                "context": ctx,
                "answer": g.get("text") if isinstance(g, dict) else g,
                "gold": golds[0] if golds else ""
            })
    return out


def _labels_em(flat_rows: List[Dict[str, Any]]) -> List[int]:
    lab = []
    for r in flat_rows:
        pred = r.get("answer") or ""
        gold = r.get("gold") or ""
        lab.append(1 if exact_match(pred, [gold]) else 0)
    return lab



def _fmt_table(aurocs: Dict[str, float], model_name: str = "LLaMA-3.1-8B", dataset_name: str = "SQuAD") -> str:
    rows = [
        ("p(true)",         aurocs.get("p(true)")),
        ("Perplexity",      aurocs.get("perplexity")),
        ("SE",              aurocs.get("se")),
        ("MARS",            aurocs.get("mars")),
        ("MARS-SE",         aurocs.get("mars-se")),
        ("CCS",             aurocs.get("ccs")),
        ("SAPLMA",          aurocs.get("saplma")),
        ("HaloScope",       aurocs.get("haloscope")),
        ("HaMI* (Ours)",    aurocs.get("hami_star")),
        ("HaMI (Ours)",     aurocs.get("hami")),

        ("P_t_uncertainty", aurocs.get("P_t_uncertainty")),
        ("P_s_uncertainty", aurocs.get("P_s_uncertainty")),
        ("P_c_uncertainty", aurocs.get("P_c_uncertainty")),
    ]
    w1 = max(len(r[0]) for r in rows) + 2
    head = f"┌{'─'*38}┐\n│  {model_name:<14}    {dataset_name:<12} │\n├{'─'*38}┤"
    body = []
    for k, v in rows:
        val = "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"
        body.append(f"│ {k:<{w1}} {val:>6} │")
    tail = f"└{'─'*38}┘"
    return "\n".join([head] + body + [tail])


def _call_step(modpath: str, func: str = "main"):
    m = import_module(modpath)
    f = getattr(m, func)
    return f()


def get_scorers():
    return [
        ("se",          run_se),
        ("mars",        run_mars),
        ("mars-se",     run_mars_se),
        ("perplexity",  run_perplexity),
        ("p(true)",     run_p_true),
        ("ccs",         run_ccs),
        ("saplma",      run_saplma),
        ("haloscope",   run_haloscope),
        ("hami",        run_hami),
        ("hami_star",   run_hami_star),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print("[main] step 1/5: process_data")
    _call_step("src.process_data")

    limit = _env_int("MULTI_LIMIT", _env_int("MAIN_LIMIT", 0))
    if limit > 0:
        path = os.path.join(args.data_dir, "squad_train.json")
        _truncate_json_list(path, limit)

    print("[main] step 2/5: multi_sample")
    _call_step("src.multi_sample")

    print("[main] step 3/5: final_select")
    _call_step("src.final_select")

    print("[main] step 4/5: random_pairs")
    _call_step("src.random_pairs")

    print("[main] step 5/5: refined_set")
    _call_step("src.refined_set")

    data = load_json(os.path.join(args.data_dir, "squad_multi.json"))
    flat = _flatten(data)
    labels = _labels_em(flat)
    scores: Dict[str, List[float]] = {}
    for name, fn in get_scorers():
        print(f"[main] scoring: {name}")
        s = fn(flat)
        scores[name.lower()] = [float(x) for x in s]

    tok_u, model_u = _get_local_model(None, None)
    scores["P_t_uncertainty"] = score_p_t(flat, tok=tok_u, model=model_u)
    scores["P_s_uncertainty"] = score_p_s(flat, tok=tok_u, model=model_u)
    scores["P_c_uncertainty"] = score_p_c(flat)

    aurocs = compute_aurocs_with_labels(scores, labels)
    table = _fmt_table(aurocs)

    print("\n=== AUROC (SQuAD) ===")
    print(table)

    os.makedirs(args.results_dir, exist_ok=True)
    save_json(os.path.join(args.results_dir, "scores.json"), {"aurocs": aurocs})
    save_text(os.path.join(args.results_dir, "table.txt"), table)
    print("[main] saved results to results/")


if __name__ == "__main__":
    main()
