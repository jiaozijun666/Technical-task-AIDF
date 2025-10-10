import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

def compute_logprob(model, tok, prompt, completion):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    labels = tok(completion, return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
    return -loss.item()

def calculate_agreement(model, q, a1, a2):
    tok = AutoTokenizer.from_pretrained(model.model_id)
    p1 = np.exp(compute_logprob(model.mdl, tok, q, a1))
    p2 = np.exp(compute_logprob(model.mdl, tok, q, a2))
    return p1 / (p1 + p2 + 1e-9)

def calculate_temperature_scaled_agreement(model, q, a1, a2, T=1.5):
    tok = AutoTokenizer.from_pretrained(model.model_id)
    lp1 = compute_logprob(model.mdl, tok, q, a1)
    lp2 = compute_logprob(model.mdl, tok, q, a2)
    z = np.array([(lp1 - lp2) / T])
    return float(1 / (1 + np.exp(-z)))

def softmax(x):
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
