#!/usr/bin/env python3
"""
Train Random Forest and Logistic Regression baselines on acoustic features.

Uses speaker-disjoint split (same as transformers, CNN) for fair comparison.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import FEATURES_DIR, OUTPUTS_DIR, RANDOM_SEED, TEST_SIZE, VAL_SIZE
from src.models.baseline import get_feature_importance, make_lr_pipeline, make_rf_pipeline
from src.utils.eval import evaluate_binary, evaluate_binary_bootstrap
from src.utils.splits import speaker_disjoint_split


def main() -> int:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    # Remove previous run so re-runs don't mix old and new results (mission reproducibility)
    out_file = OUTPUTS_DIR / "baseline_results.json"
    if out_file.exists():
        out_file.unlink()

    df = pd.read_csv(FEATURES_DIR / "acoustic_features.csv")
    if df.empty:
        print("No features. Run: python scripts/extract_features.py")
        return 1

    # Speaker-disjoint split (same convention as transformers, CNN)
    pairs = [(Path(row["path"]), int(row["label"])) for _, row in df.iterrows()]
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    tr_pairs, val_pairs, test_pairs = speaker_disjoint_split(pairs, TEST_SIZE, val_ratio, RANDOM_SEED)
    if not val_pairs:
        val_pairs = tr_pairs[: len(tr_pairs) // 10]

    def path_key(p):
        return str(Path(p).resolve())

    test_paths = {path_key(p) for p, _ in test_pairs}
    tr_paths = {path_key(p) for p, _ in tr_pairs}
    val_paths = {path_key(p) for p, _ in val_pairs}

    df["_path_key"] = df["path"].apply(lambda x: path_key(x))
    df_tr = df[df["_path_key"].isin(tr_paths)].drop(columns=["_path_key"])
    df_val = df[df["_path_key"].isin(val_paths)].drop(columns=["_path_key"])
    df_te = df[df["_path_key"].isin(test_paths)].drop(columns=["_path_key"])

    train_median = df_tr.drop(columns=["path", "label"]).median()
    X_tr = df_tr.drop(columns=["path", "label"]).fillna(train_median)
    y_tr = df_tr["label"]
    X_val = df_val.drop(columns=["path", "label"]).fillna(train_median)
    y_val = df_val["label"]
    X_te = df_te.drop(columns=["path", "label"]).fillna(train_median)
    y_te = df_te["label"]

    feature_names = list(X_tr.columns)
    print(f"Speaker-disjoint: train={len(X_tr)}, val={len(X_val)}, test={len(X_te)}")

    results = {}

    for name, pipe in [
        ("random_forest", make_rf_pipeline(random_state=RANDOM_SEED)),
        ("logistic_regression", make_lr_pipeline(random_state=RANDOM_SEED)),
    ]:
        pipe.fit(X_tr, y_tr)
        # Train metrics
        tr_pred = pipe.predict(X_tr)
        tr_prob = pipe.predict_proba(X_tr)[:, 1]
        train_metrics = evaluate_binary(y_tr.values, tr_pred, tr_prob)
        # Val metrics
        val_pred = pipe.predict(X_val)
        val_prob = pipe.predict_proba(X_val)[:, 1]
        val_metrics = evaluate_binary(y_val.values, val_pred, val_prob)
        # Test metrics
        y_pred = pipe.predict(X_te)
        y_prob = pipe.predict_proba(X_te)[:, 1]
        metrics = evaluate_binary(y_te.values, y_pred, y_prob)
        ci = evaluate_binary_bootstrap(y_te.values, y_pred, y_prob, n_bootstrap=500)
        metrics["auc_ci_95"] = ci.get("auc_ci_95", [0, 0])
        results[name] = {**metrics, "train": train_metrics, "val": val_metrics}

        imp = get_feature_importance(pipe, feature_names)
        imp_sorted = sorted(imp.items(), key=lambda x: -x[1])[:20]
        results[name]["top_features"] = imp_sorted

        joblib.dump(pipe, OUTPUTS_DIR / f"baseline_{name}.joblib")
        auc_ci = metrics.get("auc_ci_95", [0, 0])
        print(f"{name}: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, AUC={metrics['auc_roc']:.3f} (95% CI: {auc_ci[0]:.3f}-{auc_ci[1]:.3f})")

    def to_json_serializable(obj):
        if isinstance(obj, (list, tuple)):
            return [to_json_serializable(x) for x in obj]
        if hasattr(obj, "item"):
            return float(obj)
        return obj

    results_ser = {}
    for k, v in results.items():
        results_ser[k] = {kk: to_json_serializable(vv) for kk, vv in v.items() if kk not in ("top_features",)}
        results_ser[k]["top_features"] = [[str(a), float(b)] for a, b in v["top_features"]]
    with open(OUTPUTS_DIR / "baseline_results.json", "w") as f:
        json.dump(results_ser, f, indent=2)

    for name in results:
        with open(OUTPUTS_DIR / f"baseline_{name}_features.txt", "w") as f:
            for feat, score in results[name]["top_features"]:
                f.write(f"{feat}\t{score}\n")

    print(f"Results saved to {OUTPUTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
