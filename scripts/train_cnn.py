#!/usr/bin/env python3
"""
Train baseline: CNN (MLP) on selected acoustic features only.

Uses speaker-disjoint split. Same feature selection as fusion (top by univariate AUC,
drop highly correlated) so no redundancy. No mel spectrogram.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BASELINE_DROPOUT,
    BASELINE_LEARNING_RATE,
    BASELINE_WEIGHT_DECAY,
    BATCH_SIZE,
    CORR_THRESHOLD,
    EARLY_STOPPING_PATIENCE,
    FEATURES_DIR,
    MAX_EPOCHS,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    TEST_SIZE,
    TOP_ACOUSTIC_N,
    VAL_SIZE,
)
from src.models.acoustic_mlp import AcousticMLP
from sklearn.preprocessing import StandardScaler

from src.utils.eval import evaluate_binary
from src.utils.feature_selection import select_acoustic_features
from src.utils.paths import get_audio_paths_with_labels
from src.utils.splits import get_speaker, save_split, speaker_disjoint_split, SPLIT_FILENAME


def _run_training_loop(
    model, train_loader, val_loader, test_loader, device, tr_pairs, criterion, best_ckpt_path,
    lr=None, weight_decay=None,
):
    """Shared training loop: early stop on val AUC, then threshold tune and return metrics."""
    lr = lr if lr is not None else 1e-4
    wd = weight_decay if weight_decay is not None else 1e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_val_auc = 0.0
    patience_counter = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

        model.eval()
        val_pred, val_prob, val_y = [], [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                prob = torch.softmax(logits, 1)[:, 1].cpu().numpy()
                pred = logits.argmax(1).cpu().numpy()
                val_pred.extend(pred)
                val_prob.extend(prob)
                val_y.extend(y.numpy())
        val_metrics = evaluate_binary(np.array(val_y), np.array(val_pred), np.array(val_prob))
        val_auc = val_metrics["auc_roc"]
        val_acc = val_metrics["accuracy"]
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            patience_counter += 1
        if (epoch + 1) <= 3 or (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1} val_acc={val_acc:.3f} val_auc={val_auc:.3f}")
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1} (val AUC did not improve for {EARLY_STOPPING_PATIENCE} epochs)")
            break

    model.load_state_dict(torch.load(best_ckpt_path, weights_only=True))
    model.eval()

    def get_probs_and_labels(loader):
        probs, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                logits = model(x)
                prob = torch.softmax(logits, 1)[:, 1].cpu().numpy()
                probs.extend(prob)
                labels.extend(y.numpy())
        return np.array(probs), np.array(labels)

    val_prob, val_y = get_probs_and_labels(val_loader)
    best_threshold = 0.5
    best_val_f1 = 0.0
    for t in np.linspace(0.05, 0.95, 19):
        pred_t = (val_prob >= t).astype(np.int64)
        m = evaluate_binary(val_y, pred_t, val_prob)
        if m["f1"] > best_val_f1:
            best_val_f1 = m["f1"]
            best_threshold = float(t)
    print(f"Best decision threshold (val F1): {best_threshold:.3f}")

    def run_metrics(loader, threshold=best_threshold):
        probs, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                logits = model(x)
                prob = torch.softmax(logits, 1)[:, 1].cpu().numpy()
                probs.extend(prob)
                labels.extend(y.numpy())
        probs = np.array(probs)
        labels = np.array(labels)
        pred = (probs >= threshold).astype(np.int64)
        return evaluate_binary(labels, pred, probs)

    return run_metrics(train_loader), run_metrics(val_loader), run_metrics(test_loader), best_threshold


def main() -> int:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )

    pairs = get_audio_paths_with_labels(PROCESSED_DIR)
    if not pairs:
        print("No processed audio. Run: python scripts/run_preprocessing.py")
        return 1

    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    tr_pairs, val_pairs, test_pairs = speaker_disjoint_split(
        pairs, TEST_SIZE, val_ratio, RANDOM_SEED
    )
    if not val_pairs:
        val_pairs = tr_pairs[: len(tr_pairs) // 10]

    save_split(tr_pairs, val_pairs, test_pairs, OUTPUTS_DIR / SPLIT_FILENAME)

    # Class weights (shared by both branches)
    n_real = sum(1 for _, l in tr_pairs if l == 0)
    n_fake = sum(1 for _, l in tr_pairs if l == 1)
    weight_real = len(tr_pairs) / (2 * max(n_real, 1))
    weight_fake = len(tr_pairs) / (2 * max(n_fake, 1))
    class_weights = torch.tensor([weight_real, weight_fake], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_ckpt_path = OUTPUTS_DIR / "cnn_spectrogram.pt"

    # Baseline: CNN on selected acoustic features only (same selection as fusion, no redundancy)
    df_feat = pd.read_csv(FEATURES_DIR / "acoustic_features.csv")
    if df_feat.empty:
        print("No acoustic features. Run: python scripts/extract_features.py")
        return 1
    path_to_feat = {}
    for idx, row in df_feat.iterrows():
        p = Path(row["path"]).resolve()
        path_to_feat[str(p)] = idx
        path_to_feat[p.as_posix()] = idx

    def get_path_key(p):
        return str(Path(p).resolve())

    def filter_pairs(prs):
        return [(p, l) for p, l in prs if get_path_key(p) in path_to_feat]

    tr_pairs = filter_pairs(tr_pairs)
    val_pairs = filter_pairs(val_pairs)
    test_pairs = filter_pairs(test_pairs)
    if not tr_pairs or not val_pairs:
        print("No overlap between split and acoustic_features.csv paths.")
        return 1

    # Class balance (counts and per-speaker)
    def class_counts(prs):
        n_real = sum(1 for _, l in prs if l == 0)
        n_fake = sum(1 for _, l in prs if l == 1)
        return n_real, n_fake

    def speaker_label_summary(prs):
        spk_labels = {}
        for p, l in prs:
            spk = get_speaker(p)
            if spk not in spk_labels:
                spk_labels[spk] = []
            spk_labels[spk].append(l)
        real_only = sum(1 for ls in spk_labels.values() if all(x == 0 for x in ls))
        fake_only = sum(1 for ls in spk_labels.values() if all(x == 1 for x in ls))
        mixed = len(spk_labels) - real_only - fake_only
        return real_only, fake_only, mixed

    tr_r, tr_f = class_counts(tr_pairs)
    val_r, val_f = class_counts(val_pairs)
    te_r, te_f = class_counts(test_pairs)
    print(f"Speaker-disjoint: train={len(tr_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
    print(f"Class balance: train real={tr_r} fake={tr_f}  val real={val_r} fake={val_f}  test real={te_r} fake={te_f}")
    tr_ro, tr_fo, tr_m = speaker_label_summary(tr_pairs)
    val_ro, val_fo, val_m = speaker_label_summary(val_pairs)
    te_ro, te_fo, te_m = speaker_label_summary(test_pairs)
    print(f"Per-speaker: train {tr_ro} real-only, {tr_fo} fake-only, {tr_m} mixed  |  val {val_ro}/{val_fo}/{val_m}  |  test {te_ro}/{te_fo}/{te_m}")
    feat_cols = [c for c in df_feat.columns if c not in ("path", "label")]

    def get_acoustic(prs):
        return np.array([
            df_feat.iloc[path_to_feat[get_path_key(p)]][feat_cols].values for p, _ in prs
        ], dtype=np.float64)

    X_tr = get_acoustic(tr_pairs)
    X_val = get_acoustic(val_pairs)
    X_te = get_acoustic(test_pairs)
    train_median = np.nanmedian(X_tr, axis=0)
    X_tr = np.nan_to_num(X_tr, nan=train_median)
    X_val = np.nan_to_num(X_val, nan=train_median)
    X_te = np.nan_to_num(X_te, nan=train_median)

    y_tr = np.array([l for _, l in tr_pairs])
    sel_idx = select_acoustic_features(
        X_tr, y_tr, top_n=TOP_ACOUSTIC_N, corr_threshold=CORR_THRESHOLD
    )
    X_tr = X_tr[:, sel_idx]
    X_val = X_val[:, sel_idx]
    X_te = X_te[:, sel_idx]
    selected_names = [feat_cols[i] for i in sel_idx]
    n_feat = X_tr.shape[1]
    print(f"CNN on selected acoustic features (no redundancy): {n_feat}-d")

    # Standardize: fit on train, transform train/val/test
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te = scaler.transform(X_te)

    y_val = np.array([l for _, l in val_pairs])
    y_te = np.array([l for _, l in test_pairs])
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr.astype(np.float32)).float(), torch.from_numpy(y_tr).long()),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val.astype(np.float32)).float(), torch.from_numpy(y_val).long()),
        batch_size=BATCH_SIZE,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te.astype(np.float32)).float(), torch.from_numpy(y_te).long()),
        batch_size=BATCH_SIZE,
    )

    model = AcousticMLP(n_input=n_feat, n_classes=2, dropout=BASELINE_DROPOUT).to(device)
    print("Baseline: CNN on selected acoustic features")
    train_metrics, val_metrics, test_metrics, best_threshold = _run_training_loop(
        model, train_loader, val_loader, test_loader, device, tr_pairs, criterion, best_ckpt_path,
        lr=BASELINE_LEARNING_RATE, weight_decay=BASELINE_WEIGHT_DECAY,
    )
    print("Baseline Train:", train_metrics, "Val:", val_metrics, "Test:", test_metrics)

    config_path = OUTPUTS_DIR / "acoustic_baseline_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "sel_idx": [int(i) for i in sel_idx],
            "train_median": train_median.tolist(),
            "feature_names": selected_names,
        }, f, indent=2)
    joblib.dump(scaler, OUTPUTS_DIR / "acoustic_baseline_scaler.joblib")

    results = {
        "architecture": "acoustic_mlp",
        "decision_threshold": best_threshold,
        "n_features": n_feat,
        **test_metrics,
        "train": train_metrics,
        "val": val_metrics,
    }
    with open(OUTPUTS_DIR / "cnn_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
