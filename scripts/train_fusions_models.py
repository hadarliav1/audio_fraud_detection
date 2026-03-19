#!/usr/bin/env python3
"""
Train fusion models: transformer encoder + acoustic features (end-to-end).

Uses results/split.json and acoustic_features.csv.
Writes metrics to results/fusion_models_results.json.
"""

import json
import sys
from tqdm.auto import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from transformers import AutoFeatureExtractor, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    OUTPUTS_DIR,
    FEATURES_DIR,
    RANDOM_SEED,
    SAMPLE_RATE,
    LEARNING_RATE,
    BATCH_SIZE,
    MAX_EPOCHS,
    TRANSFORMER_MODELS,
)
from src.utils.eval import evaluate_binary
from src.utils.splits import load_split, SPLIT_FILENAME, summarize_split
from src.utils.dataset_fusion import FusionDataset
from src.models.transformer_fusion import TransformerFusionModel


def model_short_name(model_id: str) -> str:
    name = model_id.split("/")[-1].split(".")[0]
    for suffix in ["-base", "-tiny", "-small", "_base", "_tiny", "_small"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace("-", "_")


def train_one(model_id: str, tr_pairs, val_pairs, test_pairs, device: torch.device):
    short = model_short_name(model_id)
    print(f"\n=== Fusion training for {model_id} ({short}) ===")

    csv_path = FEATURES_DIR / "acoustic_features.csv"
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError("acoustic_features.csv is empty")

    path_to_idx = {}
    for idx, row in df.iterrows():
        p = Path(row["path"]).resolve()
        path_to_idx[str(p)] = idx
        path_to_idx[p.as_posix()] = idx

    feat_cols = [c for c in df.columns if c not in ("path", "label")]
    acoustic_dim = len(feat_cols)

    def subset(pairs):
        xs, ys = [], []
        for p, l in pairs:
            key = str(p.resolve())
            if key not in path_to_idx and p.resolve().as_posix() not in path_to_idx:
                continue
            xs.append(p)
            ys.append(l)
        return xs, ys

    tr_paths, tr_labels = subset(tr_pairs)
    val_paths, val_labels = subset(val_pairs)
    te_paths, te_labels = subset(test_pairs)

    print(
        f"Fusion split (after acoustic join): "
        f"train={len(tr_paths)}, val={len(val_paths)}, test={len(te_paths)}"
    )

    def matrix(paths):
        rows = []
        for p in paths:
            key = str(p.resolve())
            if key not in path_to_idx:
                key = p.resolve().as_posix()
            idx = path_to_idx[key]
            rows.append(df.iloc[idx][feat_cols].values.astype(np.float64))
        return np.array(rows)

    X_tr = matrix(tr_paths)
    X_val = matrix(val_paths)
    X_te = matrix(te_paths)

    train_median = np.nanmedian(X_tr, axis=0)
    X_tr = np.nan_to_num(X_tr, nan=train_median)
    X_val = np.nan_to_num(X_val, nan=train_median)
    X_te = np.nan_to_num(X_te, nan=train_median)

    scaler = StandardScaler().fit(X_tr)

    fe = AutoFeatureExtractor.from_pretrained(model_id)

    train_ds = FusionDataset(tr_paths, tr_labels, df, scaler, feat_cols, fe)
    val_ds = FusionDataset(val_paths, val_labels, df, scaler, feat_cols, fe)
    test_ds = FusionDataset(te_paths, te_labels, df, scaler, feat_cols, fe)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = TransformerFusionModel(model_id, acoustic_dim=acoustic_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    num_training_steps = MAX_EPOCHS * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    best_val_auc = 0.0
    patience = 5
    patience_counter = 0
    ckpt_path = OUTPUTS_DIR / f"fusion_{short}.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\n[{short}] Starting epoch {epoch}")
        model.train()
        loop = tqdm(train_loader, desc=f"{short} epoch {epoch}", leave=False)
        for batch in loop:
            optimizer.zero_grad()
            logits = model(
                input_values=batch["input_values"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                acoustic=batch["acoustic"].to(device),
            )
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            loop.set_postfix(loss=float(loss.detach().cpu()))

        model.eval()
        all_prob, all_y, all_pred = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    input_values=batch["input_values"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    acoustic=batch["acoustic"].to(device),
                )
                prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                y = batch["labels"].numpy()
                pred = (prob >= 0.5).astype(np.int64)
                all_prob.extend(prob)
                all_y.extend(y)
                all_pred.extend(pred)
        all_prob = np.array(all_prob)
        all_y = np.array(all_y)
        all_pred = np.array(all_pred)
        val_metrics = evaluate_binary(all_y, all_pred, all_prob)
        val_auc = val_metrics["auc_roc"]
        print(f"Epoch {epoch} val_auc={val_auc:.3f}")

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    def collect_probs(loader):
        probs, labels = [], []
        with torch.no_grad():
            for batch in loader:
                logits = model(
                    input_values=batch["input_values"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    acoustic=batch["acoustic"].to(device),
                )
                p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                probs.extend(p)
                labels.extend(batch["labels"].numpy())
        return np.array(probs), np.array(labels)

    prob_val, y_val = collect_probs(val_loader)

    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 19):
        y_pred = (prob_val >= t).astype(np.int64)
        m = evaluate_binary(y_val, y_pred, prob_val)
        if m["f1"] >= best_f1:
            best_f1 = m["f1"]
            best_t = float(t)
    print(f"Best threshold (val F1) for {short}: {best_t:.3f}")

    def metrics_at_threshold(loader, t):
        prob, y = collect_probs(loader)
        y_pred = (prob >= t).astype(np.int64)
        return evaluate_binary(y, y_pred, prob)

    train_metrics = metrics_at_threshold(train_loader, best_t)
    val_metrics = metrics_at_threshold(val_loader, best_t)
    test_metrics = metrics_at_threshold(test_loader, best_t)

    return short, {
        "decision_threshold": best_t,
        **test_metrics,
        "train": train_metrics,
        "val": val_metrics,
    }


def main() -> int:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    split_path = OUTPUTS_DIR / SPLIT_FILENAME
    loaded = load_split(split_path)
    if loaded is None:
        print("results/split.json missing; run scripts/train_cnn.py first.")
        return 1
    tr_pairs, val_pairs, test_pairs = loaded
    summarize_split(tr_pairs, val_pairs, test_pairs, OUTPUTS_DIR / "split_report.json")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    all_results = {}
    # Fusion models to train (Stage 7):
    # - HuBERT + acoustic features
    # - WavLM + acoustic features
    # - Whisper + acoustic features (bonus, heavier compute)
    FUSION_MODELS = [
        "microsoft/wavlm-base",
        "facebook/hubert-base-ls960",
        "openai/whisper-tiny",
    ]

    for model_id in FUSION_MODELS:
        short, res = train_one(model_id, tr_pairs, val_pairs, test_pairs, device)
        all_results[short] = res

    out_file = OUTPUTS_DIR / "fusion_models_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved fusion results to {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())