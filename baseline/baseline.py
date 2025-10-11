import math, os, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from typing import List, Tuple

def _device(model): 
    return model.device

def _last_token_states(entry, layer_idx: int):
    xs = entry["layers"][layer_idx]
    if len(xs) == 0:
        return None
    return np.array(xs[-1], dtype=np.float32)

def _bce_train_epoch(model, opt, x, y):
    model.train()
    bs = 128
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i+bs]
        xb = torch.tensor(x[j], dtype=torch.float32, device=_device(model))
        yb = torch.tensor(y[j], dtype=torch.float32, device=_device(model))
        p = model(xb).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(p, yb)
        opt.zero_grad(); loss.backward(); opt.step()

def _infer_logits(model, x):
    model.eval()
    with torch.inference_mode():
        xb = torch.tensor(x, dtype=torch.float32, device=_device(model))
        p = model(xb).squeeze(-1)
        return torch.sigmoid(p).detach().cpu().numpy().tolist()

def run_p_true(entries, tok, model, args):
    system = "Answer only True or False."
    template = "Question: {q}\nPrediction: {p}\nIs the prediction exactly correct? Answer True or False."
    scores, labels = [], []
    device = _device(model)
    for e in entries:
        q, p = e["q"], e["pred"]
        prompt = system + "\n" + template.format(q=q, p=p)
        ipt = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(**ipt, do_sample=False, max_new_tokens=4, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0], skip_special_tokens=True)
        y = 1.0 if "True" in text and text.rfind("True") > text.rfind("False") else 0.0
        scores.append(float(y))
        labels.append(int(e["label"]))
    return scores, labels

def _avg_nll(tok, model, prompt, answer):
    device = _device(model)
    ids = tok(prompt + answer, return_tensors="pt").to(device).input_ids
    ans_ids = tok(answer, return_tensors="pt").to(device).input_ids
    tgt = ids.clone()
    tgt[:, :-ans_ids.shape[1]] = -100
    out = model(ids, labels=tgt)
    return out.loss.item()

def run_perplexity(entries, tok, model, args):
    scores, labels = [], []
    for e in entries:
        q, p = e["q"], e["pred"]
        nll = _avg_nll(tok, model, f"Q: {q}\nA: ", p)
        scores.append(float(-nll))
        labels.append(int(e["label"]))
    return scores, labels

def _openai_client():
    from openai import OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI()

def _entails(client, a: str, b: str, model_name: str):
    m = [
        {"role":"system","content":"Answer with one token: yes or no."},
        {"role":"user","content":f"Does \"{a}\" entail \"{b}\" semantically? Answer yes or no."}
    ]
    r = client.chat.completions.create(model=model_name, messages=m, temperature=0)
    t = r.choices[0].message.content.strip().lower()
    return t.startswith("y")

def run_se(entries, tok, model, args):
    M = getattr(args, "se_samples", 6)
    client = _openai_client()
    se_model = getattr(args, "se_judge_model", "gpt-3.5-turbo")
    device = _device(model)
    scores, labels = [], []
    for e in entries:
        q = e["q"]
        gens = []
        for _ in range(M):
            ipt = tok(f"Q: {q}\nA:", return_tensors="pt").to(device)
            out = model.generate(**ipt, do_sample=True, temperature=getattr(args, "temperature", 0.5), top_p=0.95, max_new_tokens=getattr(args, "max_new_tokens", 64), eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
            gens.append(tok.decode(out[0], skip_special_tokens=True).split("A:",1)[-1].strip())
        clusters = []
        for g in gens:
            placed = False
            for c in clusters:
                if _entails(client, g, c[0], se_model) and _entails(client, c[0], g, se_model):
                    c.append(g); placed = True; break
            if not placed:
                clusters.append([g])
        main = max(len(c) for c in clusters) if clusters else 1
        pc = main / max(1, len(gens))
        scores.append(float(pc))
        labels.append(int(e["label"]))
    return scores, labels

def _content_weight(tokens: List[str]):
    ws = []
    for t in tokens:
        if t.strip() == "" or t in [",",".",";","?","!","'",'"',":","-","(",")"]:
            ws.append(0.0)
        else:
            ws.append(1.0)
    return np.array(ws, dtype=np.float32)

def run_mars(entries, tok, model, args):
    scores, labels = [], []
    for e in entries:
        q, p = e["q"], e["pred"]
        prompt = f"Q: {q}\nA:"
        device = _device(model)
        with torch.inference_mode():
            ids_q = tok(prompt, return_tensors="pt").to(device).input_ids
            ids_all = tok(prompt + " " + p, return_tensors="pt").to(device).input_ids
            loc = ids_all.shape[1] - (tok(" " + p, return_tensors="pt").input_ids.shape[1])
            seq = ids_all
            out = model(seq)
            logp = F.log_softmax(out.logits[:, :-1, :], dim=-1)
            y = seq[:, 1:]
            token_logp = torch.gather(logp, 2, y.unsqueeze(-1)).squeeze(-1)[0]
            gen_logp = token_logp[loc-1:]
            gen_toks = tok.convert_ids_to_tokens(seq[0, loc:].tolist())
            w = _content_weight(gen_toks[:gen_logp.shape[0]])
            if w.sum() == 0:
                scores.append(float(gen_logp.mean().item()))
            else:
                s = (torch.tensor(w, device=device) * gen_logp).sum() / torch.tensor(w.sum(), device=device)
                scores.append(float(s.item()))
        labels.append(int(e["label"]))
    return scores, labels

def run_mars_se(entries, tok, model, args):
    alpha = 0.5
    s1, _ = run_mars(entries, tok, model, args)
    s2, _ = run_se(entries, tok, model, args)
    scores = [float(alpha*s1[i] + (1.0-alpha)*s2[i]) for i in range(len(entries))]
    labels = [int(e["label"]) for e in entries]
    return scores, labels

class _LinearProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 1)
    def forward(self, x):
        return self.fc(x)

class _SAPLMAProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

def run_ccs(entries, tok, model, args):
    layer = getattr(args, "hidden_layer", 12)
    base = []
    for e in entries:
        v = _last_token_states(e, layer)
        if v is not None:
            base.append(v)
        else:
            base.append(np.zeros(4096, dtype=np.float32))
    base = np.stack(base)
    pert = np.copy(base)
    x = np.concatenate([base, pert], 0)
    y = np.array([int(e["label"]) for e in entries] + [int(e["label"]) for e in entries], dtype=np.float32)
    d = x.shape[1]
    probe = _LinearProbe(d).to(_device(model))
    opt = torch.optim.Adam(probe.parameters(), lr=getattr(args, "lr", 1e-3))
    for _ in range(max(1, getattr(args, "epochs", 3))):
        _bce_train_epoch(probe, opt, x, y)
    scores = _infer_logits(probe, base)
    labels = [int(e["label"]) for e in entries]
    return [float(s) for s in scores], labels

def run_saplma(entries, tok, model, args):
    layer = getattr(args, "hidden_layer", 12)
    feats, labels = [], []
    for e in entries:
        v = _last_token_states(e, layer)
        if v is not None:
            feats.append(v)
        else:
            feats.append(np.zeros(4096, dtype=np.float32))
        labels.append(int(e["label"]))
    x = np.stack(feats); y = np.array(labels, dtype=np.float32)
    d = x.shape[1]
    clf = _SAPLMAProbe(d).to(_device(model))
    opt = torch.optim.Adam(clf.parameters(), lr=getattr(args, "lr", 1e-3))
    for _ in range(max(1, getattr(args, "epochs", 3))):
        _bce_train_epoch(clf, opt, x, y)
    scores = _infer_logits(clf, x)
    return [float(s) for s in scores], labels

class _HaloProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,256), nn.ReLU(), nn.Linear(256,1))
    def forward(self, x): 
        return self.net(x)

def run_haloscope(entries, tok, model, args):
    layer = getattr(args, "hidden_layer", 12)
    feats = []
    for e in entries:
        v = _last_token_states(e, layer)
        if v is not None:
            feats.append(v)
        else:
            feats.append(np.zeros(4096, dtype=np.float32))
    x = np.stack(feats)
    x_t = torch.tensor(x, dtype=torch.float32, device=_device(model))
    with torch.inference_mode():
        mu = x_t.mean(dim=0, keepdim=True)
        cov = torch.cov((x_t - mu).T) + 1e-4*torch.eye(x_t.shape[1], device=_device(model))
        inv = torch.inverse(cov)
        m = ((x_t - mu) @ inv * (x_t - mu)).sum(-1)
        m = (m - m.min())/(m.max()-m.min()+1e-8)
        soft = 1.0 - m
        y0 = (soft > soft.median()).float().cpu().numpy()
    d = x.shape[1]
    probe = _HaloProbe(d).to(_device(model))
    opt = torch.optim.Adam(probe.parameters(), lr=getattr(args, "lr", 1e-3))
    _bce_train_epoch(probe, opt, x, y0)
    scores = _infer_logits(probe, x)
    labels = [int(e["label"]) for e in entries]
    return [float(s) for s in scores], labels

REGISTRY = {
    "p_true": run_p_true,
    "perplexity": run_perplexity,
    "se": run_se,
    "mars": run_mars,
    "mars_se": run_mars_se,
    "ccs": run_ccs,
    "saplma": run_saplma,
    "haloscope": run_haloscope,
}
