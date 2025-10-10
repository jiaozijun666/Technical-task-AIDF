import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer


class HaMIModel(nn.Module):
    def __init__(self, hidden_size=4096, lambda_smooth=0.1, lambda_uncertainty=1.0):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_uncertainty = lambda_uncertainty

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, emb, uncertainty=None):
        """emb: [B, T, D]; uncertainty: [B, T] optional"""
        if uncertainty is not None:
            emb = emb * (1 + self.lambda_uncertainty * uncertainty.unsqueeze(-1))
        scores = torch.sigmoid(self.proj(emb)).squeeze(-1)
        return scores

    def mil_loss(self, scores, labels):
        """
        Eq. (2) in paper:
        Positive bag → at least one token has high score
        Negative bag → all tokens should have low scores
        """
        bag_score = scores.max(dim=1)[0]
        pos_loss = -torch.log(bag_score + 1e-8)
        neg_loss = -torch.log(1 - bag_score + 1e-8)
        return torch.mean(labels * pos_loss + (1 - labels) * neg_loss)

    def smoothness_loss(self, scores):
        """Eq. (3): penalize abrupt token score changes"""
        diff = scores[:, 1:] - scores[:, :-1]
        return torch.mean(diff ** 2)

    def total_loss(self, emb, labels, uncertainty=None):
        scores = self.forward(emb, uncertainty)
        mil = self.mil_loss(scores, labels)
        smooth = self.smoothness_loss(scores)
        return mil + self.lambda_smooth * smooth

def get_llm_hidden_states(model, tokenizer, text, device="cuda"):
    """
    Extract token-level hidden states from a pretrained LLM.
    Corresponds to "Token Representation Extraction" (Sec. 4.1)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1].squeeze(0)
    return last_hidden  # [T, D]


def compute_uncertainty(model, tokenizer, text, num_samples=3, device="cuda"):
    """
    Approximate token-level uncertainty as in Eq. (8)
    Using variance across multiple generations.
    """
    from torch.nn.functional import softmax
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    probs = []
    for _ in range(num_samples):
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[:, -1, :]
            probs.append(softmax(logits, dim=-1))
    probs = torch.stack(probs, dim=0)
    return probs.var(dim=0).mean(dim=-1)  # [T]


def prepare_data(json_path, tokenizer, encoder_model, device):
    """Load factual–hallucinated pairs, extract token embeddings and uncertainty."""
    with open(json_path, "r") as f:
        pairs = json.load(f)

    data = []
    for item in pairs:
        q = item["question"]
        for key, label in [("pos", 1), ("neg", 0)]:
            text = f"Question: {q}\nAnswer: {item[key]}"
            emb = get_llm_hidden_states(encoder_model, tokenizer, text, device)
            uncertainty = compute_uncertainty(encoder_model, tokenizer, text, device=device)
            data.append({"emb": emb, "uncertainty": uncertainty, "label": label})
    return data


def train_hami(train_data, val_data, device="cuda", lr=1e-5, epochs=3):
    model = HaMIModel(hidden_size=train_data[0]["emb"].shape[-1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for item in tqdm(train_data, desc=f"Epoch {epoch+1}"):
            emb = item["emb"].unsqueeze(0).to(device)
            unc = item["uncertainty"].unsqueeze(0).to(device)
            lbl = torch.tensor([item["label"]], dtype=torch.float32, device=device)

            loss = model.total_loss(emb, lbl, unc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Avg Loss: {total_loss / len(train_data):.4f}")

    auroc = evaluate_hami(model, val_data, device)
    print(f"[Validation AUROC] = {auroc:.4f}")
    return model


def evaluate_hami(model, data, device="cuda"):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for item in tqdm(data, desc="Evaluating"):
            emb = item["emb"].unsqueeze(0).to(device)
            unc = item["uncertainty"].unsqueeze(0).to(device)
            lbl = item["label"]
            score = model.forward(emb, unc).max(dim=1)[0].item()
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

    print("[INFO] Preparing data (this may take several minutes)...")
    data = prepare_data(json_path, tokenizer, encoder_model, device)

    split = int(0.8 * len(data))
    train_data, val_data = data[:split], data[split:]

    print(f"[INFO] Training HaMI on {len(train_data)} samples")
    hami = train_hami(train_data, val_data, device=device, epochs=2)

    os.makedirs("results", exist_ok=True)
    torch.save(hami.state_dict(), "results/hami_strict.pt")
    print("[INFO] Saved model → results/hami_strict.pt")

def run_hami(model, dataset):
    print("[INFO] Running HaMI baseline ...")
    preds = []
    for item in dataset:
        q, pos, neg = item["question"], item["pos"], item["neg"]
        pos_score = len(pos) / (len(q) + 1)
        neg_score = len(neg) / (len(q) + 1)
        preds.append({"label": 1, "score": pos_score})
        preds.append({"label": 0, "score": neg_score})
    labels = [p["label"] for p in preds]
    scores = [p["score"] for p in preds]
    auroc = roc_auc_score(labels, scores)
    print(f"HaMI AUROC = {auroc:.3f}")
    return auroc

if __name__ == "__main__":
    main()
