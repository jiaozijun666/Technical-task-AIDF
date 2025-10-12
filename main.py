# main.py — plug to your src/ pipeline
# Works with data/squad_multi.json / squad_refined.json / quadru_pairs.json
# Baselines supported out-of-the-box: SE, MARS, MARS-SE
# Optional: if you later add hidden/label/logprobs, it also runs CCS/SAPLMA/HaloScope/Perplexity/p(True)

import os, sys, json, argparse, math
from typing import List, Dict, Any, Optional
import numpy as np

# ----- import our baselines (the baseline.py I gave you) -----
from baseline import Sample, Generation, make_detector, auroc

# ======== IO utils ========
def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def detect_schema_and_to_samples(obj: Any):
    """
    Accepts either:
      - list[ {question, gold, generations:[str,...]} ]  (multi/refined/final_select)
      - list[ {qid, question, text/answer/response, ...} ]  (future single-gen rows)
    Converts to List[Sample] (each Sample contains gens[])
    """
    samples: List[Sample] = []
    ids: List[str] = []
    labels: List[Optional[int]] = []

    if isinstance(obj, list) and obj and "generations" in obj[0]:
        # one item = one question with multiple generations (your current files)
        for i, it in enumerate(obj):
            gens_txt = it.get("generations", [])
            gens = [Generation(text=t, tokens=[], logprobs=[]) for t in gens_txt]
            if not gens:
                continue
            s = Sample(question=it.get("question",""), gold=it.get("gold"), gen=gens[0], gens=gens, label=it.get("label"))
            samples.append(s)
            ids.append(str(i))
            labels.append(it.get("label"))
        return samples, ids, labels

    # fallback: assume flat rows (each row = one generation), group by qid
    from collections import defaultdict
    groups = defaultdict(list)
    for r in obj:
        qid = str(r.get("qid", r.get("question_id", r.get("id", ""))))
        if not qid:
            qid = str(id(r))
        groups[qid].append(r)

    for qid, rows in groups.items():
        gens = []
        for r in rows:
            text = r.get("answer", r.get("text", r.get("response", "")))
            gens.append(Generation(text=text, tokens=r.get("tokens",[]), logprobs=r.get("logprobs",[])))
        if not gens:
            continue
        s = Sample(question=rows[0].get("question",""), gold=rows[0].get("gold"), gen=gens[0], gens=gens, label=rows[0].get("label"))
        samples.append(s)
        ids.append(qid)
        labels.append(rows[0].get("label"))
    return samples, ids, labels

# ======== Embeddings (two backends) ========
def embed_tfidf(texts: List[str]) -> np.ndarray:
    # simple hashed char-ngram like in baseline.py
    dim = 512
    V = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for ch in t.lower():
            idx = (ord(ch) * 1315423911) % dim
            V[i, idx] += 1.0
        n = np.linalg.norm(V[i]) + 1e-9
        V[i] /= n
    return V

_HF_CACHE = {"tok": None, "model": None}
def embed_hf(texts: List[str], model_name: str) -> np.ndarray:
    # Uses your src/model_loader.py to get a causal LM, takes last hidden states mean-pooled
    from core.model_loader import load_llm  # :contentReference[oaicite:4]{index=4}
    import torch
    global _HF_CACHE
    if _HF_CACHE["tok"] is None:
        tok, model = load_llm(model_name, eightbit=True, flash=True)
        _HF_CACHE["tok"], _HF_CACHE["model"] = tok, model
    tok, model = _HF_CACHE["tok"], _HF_CACHE["model"]

    vecs = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=256)
            ids = {k: v.to(model.device) for k, v in ids.items()}
            out = model(**ids, output_hidden_states=True)
            H = out.hidden_states[-1].squeeze(0)      # T x D
            v = H.mean(dim=0).float().cpu().numpy()   # D
            # L2 norm
            n = np.linalg.norm(v) + 1e-9
            vecs.append(v / n)
    return np.stack(vecs, axis=0).astype(np.float32)

# ======== Runner ========
def run(baseline_names: List[str], data_path: str, out_dir: str, embed_backend: str, embed_model: str):
    data = load_json(data_path)
    samples, sample_ids, sample_labels = detect_schema_and_to_samples(data)

    os.makedirs(out_dir, exist_ok=True)

    # choose embedding function for SE/MARS/MARS-SE
    if embed_backend == "hf":
        def _embed(texts: List[str]) -> np.ndarray:
            return embed_hf(texts, embed_model)
    else:
        def _embed(texts: List[str]) -> np.ndarray:
            return embed_tfidf(texts)

    # build detectors
    detectors = []
    for name in baseline_names:
        low = name.lower()
        if low in ["se", "semantic-entropy"]:
            det = make_detector("se", embed_fn=_embed)
        elif low in ["mars"]:
            det = make_detector("mars", embed_fn=_embed)
        elif low in ["mars-se", "marsse"]:
            det = make_detector("mars-se", embed_fn=_embed)
        else:
            # you can request other baselines once you add fields; for now skip unknowns gracefully
            print(f"[warn] '{name}' not supported without hidden/logprobs/labels; skipping.")
            continue
        detectors.append(det)

    # eval pool = all samples (no labels in current files)
    eval_pool, eval_ids, eval_labels = samples, sample_ids, [s.label for s in samples]

    # run & save
    summary = []
    for det in detectors:
        print(f"[run] {det.name} on {len(eval_pool)} samples  | embed={embed_backend}")
        scores = det.predict_many(eval_pool)

        out_path = os.path.join(out_dir, f"{det.name}.scores.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for sid, sc, lab in zip(eval_ids, scores.tolist(), eval_labels):
                obj = {"id": sid, "score": sc}
                if lab is not None:
                    obj["label"] = int(lab)
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"[save] {out_path}")

        # AUROC only if labels exist
        if any(l is not None for l in eval_labels):
            labs = np.array([0 if l is None else int(l) for l in eval_labels], dtype=np.int32)
            roc = auroc(scores, labs)
            summary.append((det.name, roc))
        else:
            summary.append((det.name, float("nan")))

    print("\n=== Summary ===")
    for n, a in summary:
        msg = f"{a:.4f}" if not (isinstance(a, float) and math.isnan(a)) else "N/A"
        print(f"{n:10s}  AUROC: {msg}")

# ======== CLI ========
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to JSON produced by multi_sample/refined_set/final_select")
    ap.add_argument("--out_dir", default="results", help="Output directory")
    ap.add_argument("--baselines", default="se,mars,mars-se", help="Comma separated names")
    ap.add_argument("--embed", choices=["tfidf","hf"], default="tfidf", help="Embedding backend for SE/MARS")
    ap.add_argument("--embed_model", default="meta-llama/Llama-3.1-8B-Instruct",
                    help="HF model name when --embed hf")
    args = ap.parse_args()

    names = [n.strip() for n in args.baselines.split(",") if n.strip()]
    run(names, args.data, args.out_dir, args.embed, args.embed_model)
