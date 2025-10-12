import math, numpy as np, torch, torch.nn as nn

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

def run_hami_star(entries, tok, model, args):
    device = model.device
    hl = getattr(args,"hidden_layer",0)
    rk = getattr(args,"rk",0.1)
    lr = getattr(args,"lr",1e-3)
    epochs = max(1, getattr(args,"epochs",3))
    reps = entries[0]["layers"][hl]
    d = reps.shape[-1] if len(reps)>0 else 4096
    det = Detector(d).to(device)
    opt = torch.optim.Adam(det.parameters(), lr=lr)
    for _ in range(epochs):
        pos, neg, sms = [], [], []
        for e in entries:
            xs = e["layers"][hl]
            if len(xs)==0: continue
            x = torch.tensor(np.stack(xs), dtype=torch.float32, device=device)
            s = det(x)
            sms.append(_smooth(s))
            if int(e["label"])==1: pos.append(s)
            else: neg.append(s)
            if len(pos)+len(neg) >= 16:
                pos_mean = torch.stack([_topk_mean(si, rk) for si in pos]).mean() if pos else torch.tensor(0.0, device=device)
                neg_mean = torch.stack([_topk_mean(si, rk) for si in neg]).mean() if neg else torch.tensor(0.0, device=device)
                sm = torch.stack(sms).mean() if sms else torch.tensor(0.0, device=device)
                loss = (1.0 - pos_mean)**2 + (neg_mean)**2 + sm
                opt.zero_grad(); loss.backward(); opt.step()
                pos, neg, sms = [], [], []
        if pos or neg:
            pos_mean = torch.stack([_topk_mean(si, rk) for si in pos]).mean() if pos else torch.tensor(0.0, device=device)
            neg_mean = torch.stack([_topk_mean(si, rk) for si in neg]).mean() if neg else torch.tensor(0.0, device=device)
            sm = torch.stack(sms).mean() if sms else torch.tensor(0.0, device=device)
            loss = (1.0 - pos_mean)**2 + (neg_mean)**2 + sm
            opt.zero_grad(); loss.backward(); opt.step()
    scores, labels = [], []
    for e in entries:
        xs = e["layers"][hl]
        if len(xs)==0:
            scores.append(0.0); labels.append(int(e["label"])); continue
        x = torch.tensor(np.stack(xs), dtype=torch.float32, device=device)
        s = det(x)
        l = s.numel(); k = max(1, int(math.floor(0.1*l)+1))
        scores.append(torch.topk(s,k=k,largest=True).values.mean().item())
        labels.append(int(e["label"]))
    return scores, labels

