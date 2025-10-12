import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

class Detector(nn.Module):
    def __init__(self, d=4096, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h,1), nn.Sigmoid())
    def forward(self, x):
        return self.net(x).squeeze(-1)

def _topk_mean(s, rk):
    l = s.numel()
    k = max(1, int(math.floor(rk*l)+1))
    return torch.topk(s, k=k, largest=True).values.mean()

def _smooth(s):
    if s.numel()<2: return s.new_tensor(0.0)
    d = s[1:]-s[:-1]
    return (d*d).mean()

def _avg_nll(tok, model, q, a):
    device = model.device
    ids = tok(f"Q: {q}\nA: {a}", return_tensors="pt").to(device).input_ids
    ans_ids = tok(a, return_tensors="pt").to(device).input_ids
    tgt = ids.clone()
    tgt[:, :-ans_ids.shape[1]] = -100
    out = model(ids, labels=tgt)
    return float(out.loss.item())

def run_hami(entries, tok, model, args):
    device = model.device
    hl = getattr(args,"hidden_layer",0)
    rk = getattr(args,"rk",0.1)
    lr = getattr(args,"lr",1e-3)
    epochs = max(1, getattr(args,"epochs",3))
    reps = entries[0]["layers"][hl]
    d = reps.shape[-1] if len(reps)>0 else 4096
    nlls = []
    for e in entries:
        q = e["q"]; p = e["pred"]
        nlls.append(_avg_nll(tok, model, q, p))
    nlls = np.array(nlls, dtype=np.float32)
    w = np.exp(-nlls)
    if np.max(w)-np.min(w) > 1e-8:
        w = (w - np.min(w)) / (np.max(w) - np.min(w))
    w = w.astype(np.float32)
    det = Detector(d).to(device)
    opt = torch.optim.Adam(det.parameters(), lr=lr)
    idx = np.arange(len(entries))
    for _ in range(epochs):
        np.random.shuffle(idx)
        pos_batch, neg_batch, wp, wn, sm_terms = [], [], [], [], []
        for j in idx:
            e = entries[j]
            xs = e["layers"][hl]
            if len(xs)==0: continue
            x = torch.tensor(np.stack(xs), dtype=torch.float32, device=device)
            s = det(x)
            sm_terms.append(_smooth(s))
            if int(e["label"])==1:
                pos_batch.append(s); wp.append(w[j])
            else:
                neg_batch.append(s); wn.append(1.0 - w[j])
            if len(pos_batch)+len(neg_batch) >= 16:
                if pos_batch:
                    tp = torch.stack([_topk_mean(si, rk) for si in pos_batch])
                    wp_t = torch.tensor(wp, dtype=torch.float32, device=device)
                    pos_mean = (tp*wp_t).sum()/(wp_t.sum()+1e-8)
                else:
                    pos_mean = torch.tensor(0.0, device=device)
                if neg_batch:
                    tn = torch.stack([_topk_mean(si, rk) for si in neg_batch])
                    wn_t = torch.tensor(wn, dtype=torch.float32, device=device)
                    neg_mean = (tn*wn_t).sum()/(wn_t.sum()+1e-8)
                else:
                    neg_mean = torch.tensor(0.0, device=device)
                sm = torch.stack(sm_terms).mean() if sm_terms else torch.tensor(0.0, device=device)
                loss = (1.0 - pos_mean)**2 + (neg_mean)**2 + sm
                opt.zero_grad(); loss.backward(); opt.step()
                pos_batch, neg_batch, wp, wn, sm_terms = [], [], [], [], []
        if pos_batch or neg_batch:
            if pos_batch:
                tp = torch.stack([_topk_mean(si, rk) for si in pos_batch])
                wp_t = torch.tensor(wp, dtype=torch.float32, device=device)
                pos_mean = (tp*wp_t).sum()/(wp_t.sum()+1e-8)
            else:
                pos_mean = torch.tensor(0.0, device=device)
            if neg_batch:
                tn = torch.stack([_topk_mean(si, rk) for si in neg_batch])
                wn_t = torch.tensor(wn, dtype=torch.float32, device=device)
                neg_mean = (tn*wn_t).sum()/(wn_t.sum()+1e-8)
            else:
                neg_mean = torch.tensor(0.0, device=device)
            sm = torch.stack(sm_terms).mean() if sm_terms else torch.tensor(0.0, device=device)
            loss = (1.0 - pos_mean)**2 + (neg_mean)**2 + sm
            opt.zero_grad(); loss.backward(); opt.step()
    scores, labels = [], []
    for j,e in enumerate(entries):
        xs = e["layers"][hl]
        if len(xs)==0:
            scores.append(0.0); labels.append(int(e["label"])); continue
        x = torch.tensor(np.stack(xs), dtype=torch.float32, device=device)
        s = det(x)
        l = s.numel(); k = max(1, int(math.floor(0.1*l)+1))
        b = torch.topk(s,k=k,largest=True).values.mean().item()
        scores.append(float(b * w[j]))
        labels.append(int(e["label"]))
    return scores, labels
