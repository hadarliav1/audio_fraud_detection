"""
Classical baseline models: Random Forest and Logistic Regression.
"""

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_rf_pipeline(n_estimators: int = 100, max_depth: Optional[int] = 10, random_state: int = 42) -> Pipeline:
    """Random Forest with scaling."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)),
    ])


def make_lr_pipeline(C: float = 1.0, random_state: int = 42) -> Pipeline:
    """Logistic Regression with L2 regularization and scaling."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=1000, random_state=random_state, solver="lbfgs")),
    ])


def get_feature_importance(pipe: Pipeline, feature_names: list) -> dict:
    """Extract feature importance from Random Forest or coefficients from Logistic Regression."""
    clf = pipe.named_steps["clf"]
    if isinstance(clf, RandomForestClassifier):
        imp = clf.feature_importances_
    elif isinstance(clf, LogisticRegression):
        imp = np.abs(clf.coef_[0])
    else:
        return {}
    return {name: float(imp[i]) for i, name in enumerate(feature_names)}
