#!/usr/bin/env python3
import os, json, argparse, random, csv, numpy as np, torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from core.model_loader import load_llm
from core.text_norm import exact_match
from core.eval import auroc
from baseline.baseline import REGISTRY as BASELINES
from HaMI.hami import run as run_hami
from HaMI.hami_star import run as run_hami_star

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

def load_squad(n_train, n_test, seed):
    ds = load_dataset("squad")
    data = []
    for r in ds["train"]:
        q = r["question"]
        g = [a for a in r["answers"]["text"]]
        data.append({"q": q, "g": g})
    random.Random(seed).shuffle(data)
    train_pairs = data[:n_train]
    test_pairs = data[n_train:n_train+n_test]
    return train_pairs, test_pairs

@torch.inference_mode()
def generate_once(tok, model, q, max_new_tokens):
    prompt = f"Q: {q}\nA:"
    device = model.device
    ipt = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**ipt, do_sample=True, temperature=0.5, top_p=0.95, max_new_tokens=max_new_tokens, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, return_dict_in_generate=True)
    text = tok.decode(out.sequences[0], skip_special_tokens=True).split("A:", 1)[-1].strip()
    seq = out.sequences[0].unsqueeze(0)
    hs = model(input_ids=seq, output_hidden_states=True, use_cache=False).hidden_states
    gen_len = seq.shape[1] - ipt.input_ids.shape[1]
    rep = [h[0, -gen_len:, :].float().cpu().numpy() if gen_len>0 else np.zeros((0, h.size(-1)), dtype=np.float32) for h in hs]
    return text, rep

def prepare_entries(pairs, tok, model, max_new_tokens):
    entries = []
    for item in pairs:
        pred, layers = generate_once(tok, model, item["q"], max_new_tokens)
        lbl = int(exact_match(pred, item["g"]))
        entries.append({"q": item["q"], "g": item["g"], "pred": pred, "label": lbl, "layers": layers})
    return entries

def run_baselines(entries, tok, model, methods, args):
    out = {}
    for m in methods:
        scores, labels = BASELINES[m](entries, tok, model, args)
        out[m] = {"auroc": float(auroc(scores, labels)), "scores": [float(s) for s in scores], "labels": [int(x) for x in labels]}
    return out

def run_hami_family(entries, tok, model, args):
    res = {}
    scores, labels = run_hami(entries, tok, model, args)
    res["hami"] = {"auroc": float(auroc(scores, labels)), "scores": [float(s) for s in scores], "labels": [int(x) for x in labels]}
    scores2, labels2 = run_hami_star(entries, tok, model, args)
    res["hami_star"] = {"auroc": float(auroc(scores2, labels2)), "scores": [float(s) for s in scores2], "labels": [int(x) for x in labels2]}
    return res

def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run"])
    p.add_argument("--dataset", default="squad")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--no_flash", action="store_true")
    p.add_argument("--hidden_layer", type=int, default=12)
    p.add_argument("--rk", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--n_train", type=int, default=2000)
    p.add_argument("--n_test", type=int, default=800)
    p.add_argument("--methods", nargs="+", default=["p_true","perplexity","se","mars","mars_se","ccs","saplma","haloscope"])
    args = p.parse_args()
    if args.cmd == "run":
        train_pairs, test_pairs = load_squad(args.n_train, args.n_test, args.seed)
        tok, model = load_llm(args.model, eightbit=(not args.fp16), flash=(not args.no_flash))
        entries = prepare_entries(test_pairs, tok, model, args.max_new_tokens)
        results = run_baselines(entries, tok, model, args.methods, args)
        results.update(run_hami_family(entries, tok, model, args))
        out_json = RESULTS / f"{args.dataset}.all_baselines.{args.model.replace('/','_')}.json"
        with open(out_json, "w") as f:
            json.dump({"dataset":args.dataset,"base_model":args.model,"n":len(entries),"hidden_layer":args.hidden_layer,"results":results,"args":vars(args)}, f)
        out_csv = RESULTS / f"auroc_{args.dataset}_{args.model.replace('/','_')}.csv"
        rows = [("method","auroc")] + [(k, results[k]["auroc"]) for k in sorted(results.keys())]
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerows(rows)
        for k in sorted(results.keys()):
            print(f"{k}\t{results[k]['auroc']:.6f}")
        print(json.dumps({"saved_json": str(out_json), "saved_csv": str(out_csv)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
