#!/usr/bin/env python3
"""
Fusion A/B: frozen HuBERT embeddings + optional acoustic features, LR on top.

No fine-tuning — encoder used as feature extractor only. Runs three ablations:
HF-only (768-d), acoustic-only (selected), fusion (768 + selected). Acoustic features
are selected by top univariate AUC and removal of highly correlated (>0.85) features.
Same split and preprocessing across all for controlled comparison.
"""

import json
import sys
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    FEATURES_DIR,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    SAMPLE_RATE,
    TEST_SIZE,
    VAL_SIZE,
)
from src.utils.eval import evaluate_binary, evaluate_binary_bootstrap
from src.utils.paths import get_audio_paths_with_labels
from src.utils.splits import speaker_disjoint_split

# HuBERT was best in model comparison; use frozen encoder here
BEST_TRANSFORMER = "facebook/hubert-base-ls960"
TRANSFORMER_CKPT = OUTPUTS_DIR / "transformer_hubert_base_ls960"
MAX_AUDIO_SAMPLES = int(5.0 * SAMPLE_RATE)
LR_C = 0.1
USE_FROZEN = True

# Feature selection: top by AUC, then drop highly correlated
TOP_ACOUSTIC_N = 30
CORR_THRESHOLD = 0.85


def select_acoustic_features(
    X: np.ndarray, y: np.ndarray, top_n: int = TOP_ACOUSTIC_N, corr_threshold: float = CORR_THRESHOLD
) -> np.ndarray:
    """Select acoustic features: top by univariate AUC, then remove highly correlated."""
    n_feats = X.shape[1]
    aucs = []
    for j in range(n_feats):
        col = X[:, j]
        if np.std(col) < 1e-9 or np.any(np.isnan(col)):
            aucs.append(0.5)
            continue
        auc = roc_auc_score(y, col)
        aucs.append(max(auc, 1 - auc))  # symmetry: either direction is discriminative
    aucs = np.array(aucs)
    top_idx = np.argsort(aucs)[::-1][:top_n]

    # Remove highly correlated: keep first of each correlated pair
    X_top = X[:, top_idx]
    corr = np.corrcoef(X_top.T)
    np.fill_diagonal(corr, 0)
    keep = []
    for i in range(len(top_idx)):
        if all(np.abs(corr[i, j]) < corr_threshold for j in keep):
            keep.append(i)
    return top_idx[keep]


def _get_encoder(model):
    """Get encoder submodule (wav2vec2, hubert, wavlm, etc.)."""
    for attr in ("hubert", "wav2vec2", "wavlm"):
        enc = getattr(model, attr, None)
        if enc is not None:
            return enc
    raise ValueError("Unknown model type: no encoder found")


def extract_embeddings(model, feature_extractor, paths, device, batch_size=16):
    """Mean-pool last_hidden_state over non-padding frames."""
    model.eval()
    encoder = _get_encoder(model)
    embeddings = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        arrays = []
        for p in batch_paths:
            y, _ = librosa.load(str(p), sr=SAMPLE_RATE, mono=True)
            if len(y) > MAX_AUDIO_SAMPLES:
                start = (len(y) - MAX_AUDIO_SAMPLES) // 2
                y = y[start : start + MAX_AUDIO_SAMPLES]
            elif len(y) < MAX_AUDIO_SAMPLES:
                y = np.pad(y, (0, MAX_AUDIO_SAMPLES - len(y)), mode="constant")
            arrays.append(y.astype(np.float32))
        inputs = feature_extractor(
            arrays, sampling_rate=SAMPLE_RATE, return_tensors="pt",
            padding=True, truncation=True, max_length=MAX_AUDIO_SAMPLES
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = encoder(**inputs)
            hidden = out.last_hidden_state
            mask = inputs.get("attention_mask")
            if mask is not None:
                mask = mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            else:
                pooled = hidden.mean(1)
        embeddings.append(pooled.cpu().numpy())
    return np.vstack(embeddings)


def run_experiment(name: str, X_tr: np.ndarray, X_val: np.ndarray, X_te: np.ndarray,
                   y_tr: np.ndarray, y_val: np.ndarray, y_te: np.ndarray) -> dict:
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)

    clf = LogisticRegression(
        C=LR_C,
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    clf.fit(X_tr_s, y_tr)

    def eval_set(X, y, bootstrap=False):
        pred = clf.predict(X)
        prob = clf.predict_proba(X)[:, 1]
        m = evaluate_binary(y, pred, prob)
        if bootstrap and len(y) > 10:
            ci = evaluate_binary_bootstrap(y, pred, prob, n_bootstrap=500)
            m["auc_ci_95"] = ci.get("auc_ci_95", [0, 0])
        return m

    return {
        "name": name,
        "train": eval_set(X_tr_s, y_tr),
        "val": eval_set(X_val_s, y_val),
        "test": eval_set(X_te_s, y_te, bootstrap=True),
    }


def main() -> int:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load data, same split
    pairs = get_audio_paths_with_labels(PROCESSED_DIR)
    if not pairs:
        print("No processed audio. Run: python scripts/run_preprocessing.py")
        return 1

    df_feat = pd.read_csv(FEATURES_DIR / "acoustic_features.csv")
    if df_feat.empty:
        print("No acoustic features. Run: python scripts/extract_features.py")
        return 1

    path_to_feat = {}
    for idx, row in df_feat.iterrows():
        p = Path(row["path"]).resolve()
        path_to_feat[str(p)] = idx
        path_to_feat[p.as_posix()] = idx

    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    tr_pairs, val_pairs, test_pairs = speaker_disjoint_split(
        pairs, TEST_SIZE, val_ratio, RANDOM_SEED
    )
    if not val_pairs:
        val_pairs = tr_pairs[: len(tr_pairs) // 10]

    def get_path_key(path):
        return str(Path(path).resolve())

    def filter_pairs(pairs):
        return [(p, l) for p, l in pairs if get_path_key(p) in path_to_feat]

    tr_pairs = filter_pairs(tr_pairs)
    val_pairs = filter_pairs(val_pairs)
    test_pairs = filter_pairs(test_pairs)
    print(f"Split: train={len(tr_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")

    feat_cols = [c for c in df_feat.columns if c not in ("path", "label")]
    y_tr = np.array([l for _, l in tr_pairs])
    y_val = np.array([l for _, l in val_pairs])
    y_te = np.array([l for _, l in test_pairs])

    # 2. Load best transformer (HuBERT) and extract embeddings
    print(f"Loading {BEST_TRANSFORMER}...")
    if USE_FROZEN:
        model = AutoModelForAudioClassification.from_pretrained(BEST_TRANSFORMER, num_labels=2)
        feature_extractor = AutoFeatureExtractor.from_pretrained(BEST_TRANSFORMER)
    else:
        if TRANSFORMER_CKPT.exists():
            model = AutoModelForAudioClassification.from_pretrained(str(TRANSFORMER_CKPT))
            feature_extractor = AutoFeatureExtractor.from_pretrained(str(TRANSFORMER_CKPT))
        else:
            model = AutoModelForAudioClassification.from_pretrained(BEST_TRANSFORMER, num_labels=2)
            feature_extractor = AutoFeatureExtractor.from_pretrained(BEST_TRANSFORMER)
    model = model.to(device)

    tr_paths = [p for p, _ in tr_pairs]
    val_paths = [p for p, _ in val_pairs]
    te_paths = [p for p, _ in test_pairs]

    tr_emb = extract_embeddings(model, feature_extractor, tr_paths, device)
    val_emb = extract_embeddings(model, feature_extractor, val_paths, device)
    te_emb = extract_embeddings(model, feature_extractor, te_paths, device)
    print(f"Embeddings: {tr_emb.shape[1]}-d")

    # 3. Acoustic features (imputation only; scaling per experiment)
    def get_acoustic(pairs):
        return np.array([
            df_feat.iloc[path_to_feat[get_path_key(p)]][feat_cols].values
            for p, _ in pairs
        ], dtype=np.float64)

    X_tr_ac = get_acoustic(tr_pairs)
    X_val_ac = get_acoustic(val_pairs)
    X_te_ac = get_acoustic(test_pairs)
    train_median = np.nanmedian(X_tr_ac, axis=0)
    X_tr_ac = np.nan_to_num(X_tr_ac, nan=train_median)
    X_val_ac = np.nan_to_num(X_val_ac, nan=train_median)
    X_te_ac = np.nan_to_num(X_te_ac, nan=train_median)
    print(f"Acoustic features (raw): {X_tr_ac.shape[1]}-d")

    # Feature selection: top by AUC, drop correlated
    sel_idx = select_acoustic_features(X_tr_ac, y_tr, top_n=TOP_ACOUSTIC_N, corr_threshold=CORR_THRESHOLD)
    X_tr_ac = X_tr_ac[:, sel_idx]
    X_val_ac = X_val_ac[:, sel_idx]
    X_te_ac = X_te_ac[:, sel_idx]
    print(f"Acoustic features (selected): {X_tr_ac.shape[1]}-d")

    # 4. Run three experiments
    results = []

    # Exp 1: HuBERT embeddings only (best transformer)
    r1 = run_experiment("HuBERT Embeddings Only", tr_emb, val_emb, te_emb, y_tr, y_val, y_te)
    results.append(r1)

    # Exp 2: Acoustic only
    r2 = run_experiment("Acoustic Features Only", X_tr_ac, X_val_ac, X_te_ac, y_tr, y_val, y_te)
    results.append(r2)

    # Exp 3: Fusion
    X_tr_fuse = np.hstack([tr_emb, X_tr_ac])
    X_val_fuse = np.hstack([val_emb, X_val_ac])
    X_te_fuse = np.hstack([te_emb, X_te_ac])
    r3 = run_experiment("Fusion (HuBERT + Acoustic)", X_tr_fuse, X_val_fuse, X_te_fuse, y_tr, y_val, y_te)
    results.append(r3)

    # 5. Report
    print("\n" + "=" * 80)
    print("FUSION A/B EXPERIMENT — LogisticRegression (C={})".format(LR_C))
    print("=" * 80)

    rows = []
    for r in results:
        tr_auc = r["train"]["auc_roc"]
        val_auc = r["val"]["auc_roc"]
        te_auc = r["test"]["auc_roc"]
        te_f1 = r["test"]["f1"]
        rows.append({
            "Experiment": r["name"],
            "Train AUC": tr_auc,
            "Val AUC": val_auc,
            "Test AUC": te_auc,
            "Test F1": te_f1,
        })
        print(f"\n{r['name']}")
        print(f"  Train AUC: {tr_auc:.4f}  Val AUC: {val_auc:.4f}  Test AUC: {te_auc:.4f}  Test F1: {te_f1:.4f}")

    # 6. Analysis
    hf_te = r1["test"]["auc_roc"]
    ac_te = r2["test"]["auc_roc"]
    fusion_te = r3["test"]["auc_roc"]
    delta_hf = fusion_te - hf_te
    delta_ac = fusion_te - ac_te

    print("\n" + "-" * 80)
    print("ANALYSIS")
    print("-" * 80)
    print(f"Test AUC: HuBERT={hf_te:.4f}  Acoustic={ac_te:.4f}  Fusion={fusion_te:.4f}")
    print(f"Fusion vs HuBERT:  {delta_hf:+.4f} (|Δ|>0.01 = meaningful)")
    print(f"Fusion vs Acoustic: {delta_ac:+.4f}")

    if delta_hf > 0.01:
        conclusion = "Fusion adds orthogonal signal beyond HuBERT. KEEP FUSION."
    elif delta_hf >= -0.01:
        conclusion = "Fusion ≈ HuBERT. Transformer likely encodes most acoustic info. FUSION MARGINAL."
    else:
        conclusion = "Fusion < HuBERT. Acoustic features introduce noise/redundancy. DROP FUSION."

    print(f"\nCONCLUSION: {conclusion}")

    # 7. Save
    out = {
        "experiments": results,
        "summary": rows,
        "analysis": {
            "test_auc_hf": hf_te,
            "test_auc_acoustic": ac_te,
            "test_auc_fusion": fusion_te,
            "fusion_minus_hf": delta_hf,
            "conclusion": conclusion,
            "selected_acoustic_dim": int(X_tr_ac.shape[1]),
        },
    }
    with open(OUTPUTS_DIR / "fusion_ab_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved to {OUTPUTS_DIR / 'fusion_ab_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
