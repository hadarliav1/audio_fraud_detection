"""
Evaluation utilities for classification.
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def evaluate_binary_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Bootstrap 95% CI for AUC and F1."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs, f1s = [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_t = y_true[idx]
        y_p = y_pred[idx]
        f1s.append(float(f1_score(y_t, y_p, zero_division=0)))
        if y_prob is not None and len(np.unique(y_t)) == 2:
            y_pr = y_prob[idx]
            fpr, tpr, _ = roc_curve(y_t, y_pr)
            aucs.append(float(auc(fpr, tpr)))
    out = {}
    if aucs:
        out["auc_ci_95"] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
    if f1s:
        out["f1_ci_95"] = [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))]
    return out


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict:
    """Compute accuracy, precision, recall, F1, AUC-ROC, confusion matrix."""
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        out["auc_roc"] = float(auc(fpr, tpr))
    else:
        out["auc_roc"] = 0.0
    return out
