import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def masked_bce(prediction, target, valid):
    if not valid.any():
        raise ValueError("No valid targets")
    return torch.nn.functional.binary_cross_entropy(prediction[valid], target[valid].float())


def metrics(target, prediction):
    y, p = np.asarray(target), np.asarray(prediction, dtype=np.float64)
    if not len(y) or y.shape != p.shape or not np.isfinite(p).all():
        raise ValueError("Invalid evaluation arrays")
    if not np.isin(y, [0, 1]).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Binary labels and probabilities required")
    clipped = np.clip(p, 1e-7, 1-1e-7)
    return {"auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "auc_reason": None if len(np.unique(y)) == 2 else "single_class_targets",
            "accuracy": float(accuracy_score(y, p >= .5)),
            "f1": float(f1_score(y, p >= .5, zero_division=0)),
            "mse": float(np.mean((y-p)**2)),
            "bce": float(np.mean(-y*np.log(clipped)-(1-y)*np.log(1-clipped))),
            "n_targets": len(y)}
