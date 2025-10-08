import numpy as np

def evaluate(model, dataset):
    """
    Baseline: Perplexity
    Uses model negative log-likelihood as confidence measure.
    """
    results = []
    for item in dataset:
        q = item.get("question", "")
        pos = item.get("gold") or item.get("pos")
        neg = item.get("neg") or item.get("answer2")

        if not pos or not neg:
            continue

        try:
            ppl_true = -model.nll(q, pos)
            ppl_false = -model.nll(q, neg)
        except Exception as e:
            print(f"[Perplexity] Error: {e}")
            continue

        results.append({"label": 1, "score": ppl_true})
        results.append({"label": 0, "score": ppl_false})

    return results
