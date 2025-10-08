import numpy as np

def evaluate(model, dataset):
    """
    Baseline: p(True)
    Computes the log-probability of the model predicting the correct answer.
    """
    results = []
    for item in dataset:
        q = item.get("question", "")
        pos = item.get("gold") or item.get("pos")
        neg = item.get("neg") or item.get("answer2")

        if not pos or not neg:
            continue

        try:
            p_true = -model.nll(q, pos)
            p_false = -model.nll(q, neg)
        except Exception as e:
            print(f"[p_true] Error: {e}")
            continue

        results.append({"label": 1, "score": p_true})
        results.append({"label": 0, "score": p_false})

    return results

