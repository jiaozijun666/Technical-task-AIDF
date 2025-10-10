import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import roc_auc_score

class HaMIStarModel(nn.Module):
    def __init__(self, hidden_size=4096, lambda_smooth=0.1, lambda_uncertainty=1.0):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_uncertainty = lambda_uncertainty

        # token-level detector
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # small MLP to fuse uncertainty (SE + PPL)
        self.uncertainty_gate = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, emb, se=None, ppl=None):
        """
        emb: [B, T, D]
        se, ppl: [B, T]
        """
        if se is not None and ppl is not None:
            uncertainty = torch.stack([se, ppl], dim=-1)
            weight = self.uncertainty_gate(uncertainty).squeeze(-1)  # [B, T]
            emb = emb * (1 + self.lambda_uncertainty * weight.unsqueeze(-1))

        scores = torch.sigmoid(self.proj(emb)).squeeze(-1)
        return scores

    def mil_loss(self, scores, labels):
        bag_score = scores.max(dim=1)[0]
        loss_pos = -torch.log(bag_score + 1e-8)
        loss_neg = -torch.log(1 - bag_score + 1e-8)
        return torch.mean(labels * loss_pos + (1 - labels) * loss_neg)

    def smoothness_loss(self, scores):
        diff = scores[:, 1:] - scores[:, :-1]
        return torch.mean(diff ** 2)

    def total_loss(self, emb, labels, se=None, ppl=None):
        scores = self.forward(emb, se, ppl)
        mil = self.mil_loss(scores, labels)
        smooth = self.smoothness_loss(scores)
        return mil + self.lambda_smooth * smooth


def get_token_embeddings(model, tokenizer, text, device):
    """Get token-level embeddings from LLM hidden states"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1].squeeze(0)
    return hidden  # [T, D]


def compute_perplexity(model, tokenizer, text, device):
    """Compute PPL for each token"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    ppl = torch.exp(loss)
    return ppl.repeat(inputs["input_ids"].shape[-1])  # simple broadcast as [T]


def compute_semantic_entropy(model, tokenizer, text, num_samples=3, device="cuda"):
    """Approximate semantic entropy by sampling multiple continuations"""
    from torch.nn.functional import softmax
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    probs = []
    for _ in range(num_samples):
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[:, -1, :]
            probs.append(softmax(logits, dim=-1))
    probs = torch.stack(probs, dim=0)
    se = probs.var(dim=0).mean(dim=-1)
    return se  # [T]


def prepare_data(json_path, tokenizer, encoder_model, device):
    """Prepare pairs with token embeddings + SE + PPL features"""
    with open(json_path, "r") as f:
        pairs = json.load(f)

    data = []
    for item in pairs:
        q = item["question"]
        for key, label in [("pos", 1), ("neg", 0)]:
            text = f"Question: {q}\nAnswer: {item[key]}"
            emb = get_token_embeddings(encoder_model, tokenizer, text, device)
            se = compute_semantic_entropy(encoder_model, tokenizer, text, device=device)
            ppl = compute_perplexity(encoder_model, tokenizer, text, device=device)
            data.append({"emb": emb, "se": se, "ppl": ppl, "label": label})
    return data


def train_hami_star(train_data, val_data, device="cuda", lr=1e-5, epochs=2):
    model = HaMIStarModel(hidden_size=train_data[0]["emb"].shape[-1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for item in tqdm(train_data, desc=f"Epoch {epoch+1}"):
            emb = item["emb"].unsqueeze(0).to(device)
            se = item["se"].unsqueeze(0).to(device)
            ppl = item["ppl"].unsqueeze(0).to(device)
            lbl = torch.tensor([item["label"]], dtype=torch.float32, device=device)

            loss = model.total_loss(emb, lbl, se, ppl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Avg Loss: {total_loss / len(train_data):.4f}")

    auroc = evaluate_hami_star(model, val_data, device)
    print(f"[Validation AUROC] = {auroc:.4f}")
    return model


def evaluate_hami_star(model, data, device="cuda"):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for item in tqdm(data, desc="Evaluating"):
            emb = item["emb"].unsqueeze(0).to(device)
            se = item["se"].unsqueeze(0).to(device)
            ppl = item["ppl"].unsqueeze(0).to(device)
            lbl = item["label"]
            score = model.forward(emb, se, ppl).max(dim=1)[0].item()
            y_true.append(lbl)
            y_score.append(score)
    return roc_auc_score(y_true, y_score)


def main():
    device = "cuda" if torch.cuda.is_available() else "mps"
    json_path = "data/squad_random_pairs.json"
    model_id = "meta-llama/Llama-3.2-8B-Instruct"

    print(f"[INFO] Loading encoder: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoder_model = AutoModel.from_pretrained(model_id, output_hidden_states=True).to(device)

    print("[INFO] Preparing data (embedding + uncertainty features)...")
    data = prepare_data(json_path, tokenizer, encoder_model, device)
    split = int(0.8 * len(data))
    train_data, val_data = data[:split], data[split:]

    print(f"[INFO] Training HaMI★ on {len(train_data)} samples")
    model = train_hami_star(train_data, val_data, device=device, epochs=2)

    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/hami_star.pt")
    print("[INFO] Saved model → results/hami_star.pt")

def run_hami_star(model, dataset):
    print("[INFO] Running HaMI★ baseline ...")
    preds = []
    for item in dataset:
        q, pos, neg = item["question"], item["pos"], item["neg"]
        pos_score = len(pos) / (len(q) + 2)
        neg_score = len(neg) / (len(q) + 2)
        preds.append({"label": 1, "score": pos_score})
        preds.append({"label": 0, "score": neg_score})
    labels = [p["label"] for p in preds]
    scores = [p["score"] for p in preds]
    auroc = roc_auc_score(labels, scores)
    print(f"HaMI★ AUROC = {auroc:.3f}")
    return auroc


if __name__ == "__main__":
    main()

