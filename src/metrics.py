import numpy as np
from sklearn.metrics import roc_auc_score

def compute_auroc(scores_dict):
    summary = {}
    for k, v in scores_dict.items():
        try:
            arr = np.array(v)
            y_true = np.array([1 if i < len(arr)/2 else 0 for i in range(len(arr))])
            auroc = roc_auc_score(y_true, arr)
            summary[k] = round(float(auroc), 4)
        except Exception:
            summary[k] = "FAILED"
    return summary
