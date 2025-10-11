import re, string

def normalize_text(s: str) -> str:
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def exact_match(pred: str, golds):
    p = normalize_text(pred)
    for g in golds:
        if p == normalize_text(g):
            return 1
    return 0
