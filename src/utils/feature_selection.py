"""
Select acoustic features: top by univariate AUC, then drop highly correlated.
Shared by train_cnn (acoustic baseline) and experiment_fusion_ab (fusion).
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def select_acoustic_features(
    X: np.ndarray,
    y: np.ndarray,
    top_n: int = 30,
    corr_threshold: float = 0.85,
) -> np.ndarray:
    """
    Return indices of selected features: top_n by univariate AUC, then remove
    highly correlated (keep first of each correlated pair). No redundancy.
    """
    n_feats = X.shape[1]
    aucs = []
    for j in range(n_feats):
        col = X[:, j]
        if np.std(col) < 1e-9 or np.any(np.isnan(col)):
            aucs.append(0.5)
            continue
        auc = roc_auc_score(y, col)
        aucs.append(max(auc, 1 - auc))
    aucs = np.array(aucs)
    top_idx = np.argsort(aucs)[::-1][:top_n]

    X_top = X[:, top_idx]
    corr = np.corrcoef(X_top.T)
    np.fill_diagonal(corr, 0)
    keep = []
    for i in range(len(top_idx)):
        if all(np.abs(corr[i, j]) < corr_threshold for j in keep):
            keep.append(i)
    return top_idx[keep]
