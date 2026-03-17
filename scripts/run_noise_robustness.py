#!/usr/bin/env python3
"""
Stage 8 — Noise Robustness: Train and evaluate models on noisy data.

Pipeline:
1. Create noisy dataset (data/noisy/{noise_type}_{snr}dB/) — same split as Stage 4-7
2. Train base transformers (HuBERT, WavLM) on each noisy condition
3. Train fusion models (HuBERT+acoustic, WavLM+acoustic) on each noisy condition
4. Evaluate all models; compare clean vs noisy; base vs fusion
5. Generate plots and analysis

Output: results/noise_robustness.json, reports/noise_robustness_*.png
"""

import json
import sys
from pathlib import Path

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BATCH_SIZE,
    FEATURES_DIR,
    LEARNING_RATE,
    MAX_EPOCHS,
    NOISY_DATA_DIR,
    NOISE_TYPES,
    OUTPUTS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    SAMPLE_RATE,
    SNR_LEVELS_DB,
    SNR_LEVELS_NOISY,
)
from scripts.extract_features import extract_features_from_dir
from src.models.transformer_fusion import TransformerFusionModel
from src.utils.dataset_fusion import FusionDataset
from src.utils.eval import evaluate_binary
from src.utils.noise_utils import create_noisy_dataset, get_noisy_subdir_name
from src.utils.splits import load_split, SPLIT_FILENAME

REPORTS_DIR = PROJECT_ROOT / "reports"

# Models to compare: 2 best base + 2 fusion (HuBERT, WavLM)
BASE_MODELS = ["facebook/hubert-base-ls960", "microsoft/wavlm-base"]


def model_short_name(model_id: str) -> str:
    name = model_id.split("/")[-1].split(".")[0]
    for suffix in ["-base", "-tiny", "-small", "_base", "_tiny", "_small"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace("-", "_")


def ensure_noisy_datasets(tr_pairs, val_pairs, test_pairs) -> dict:
    """Create noisy datasets if not present. Return metadata for each condition."""
    NOISY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata_dir = NOISY_DATA_DIR / "metadata"
    all_meta = {}

    for noise_type in NOISE_TYPES:
        for snr_db in SNR_LEVELS_NOISY:
            key = get_noisy_subdir_name(noise_type, snr_db)
            meta_path = metadata_dir / f"{key}.json"

            if meta_path.exists():
                with open(meta_path) as f:
                    all_meta[key] = json.load(f)
                continue

            root, meta = create_noisy_dataset(
                tr_pairs, val_pairs, test_pairs,
                noise_type, snr_db, NOISY_DATA_DIR, SAMPLE_RATE, RANDOM_SEED,
            )
            metadata_dir.mkdir(exist_ok=True)
            data = {"root": str(root), "metadata": meta}
            with open(meta_path, "w") as f:
                json.dump(data, f, indent=2)
            all_meta[key] = data

    return all_meta


def pairs_from_metadata(meta: dict, orig_pairs: list, split: str) -> list:
    """Build (path, label) pairs for noisy data from metadata.
    Metadata order matches orig_pairs (created in same order).
    """
    noisy_list = meta["metadata"][split]  # [(orig_path, noisy_path), ...]
    return [(Path(noisy_path), label) for (_, noisy_path), (_, label) in zip(noisy_list, orig_pairs)]


def train_base_transformer(
    model_id: str,
    train_paths: list,
    train_labels: list,
    val_paths: list,
    val_labels: list,
    out_dir: Path,
    device,
) -> dict:
    """Train base transformer on given paths. Returns best threshold and val metrics."""
    short = model_short_name(model_id)
    max_samples = int(5.0 * SAMPLE_RATE)

    def load_audio_batch(paths):
        arrays = []
        for p in paths:
            y, _ = librosa.load(str(p), sr=SAMPLE_RATE, mono=True)
            if len(y) > max_samples:
                start = (len(y) - max_samples) // 2
                y = y[start : start + max_samples]
            elif len(y) < max_samples:
                y = np.pad(y, (0, max_samples - len(y)), mode="constant")
            arrays.append(y.astype(np.float32))
        return arrays

    train_ds = Dataset.from_dict({"path": train_paths, "label": train_labels})
    val_ds = Dataset.from_dict({"path": val_paths, "label": val_labels})

    fe = AutoFeatureExtractor.from_pretrained(model_id)
    max_length = int(5.0 * getattr(fe, "sampling_rate", SAMPLE_RATE))

    def preprocess(examples):
        arrays = load_audio_batch(examples["path"])
        return fe(
            arrays,
            sampling_rate=getattr(fe, "sampling_rate", SAMPLE_RATE),
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

    enc_train = train_ds.map(preprocess, remove_columns="path", batched=True, desc="Preprocess")
    enc_val = val_ds.map(preprocess, remove_columns="path", batched=True, desc="Preprocess")

    model = AutoModelForAudioClassification.from_pretrained(
        model_id, num_labels=2, label2id={"real": 0, "fake": 1}, id2label={0: "real", 1: "fake"}
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            warmup_ratio=0.1,
            logging_steps=20,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_total_limit=1,
            seed=RANDOM_SEED,
        ),
        train_dataset=enc_train,
        eval_dataset=enc_val,
        processing_class=fe,
        compute_metrics=lambda p: {
            "accuracy": float(np.mean(np.argmax(p.predictions, 1) == p.label_ids)),
            "f1": float(
                evaluate_binary(
                    p.label_ids,
                    np.argmax(p.predictions, 1),
                    (np.exp(p.predictions) / np.exp(p.predictions).sum(1, keepdims=True))[:, 1],
                )["f1"]
            ),
        },
    )

    trainer.train()
    trainer.save_model(str(out_dir))

    pred_val = trainer.predict(enc_val)
    prob_val = np.exp(pred_val.predictions) / np.exp(pred_val.predictions).sum(1, keepdims=True)
    prob_val = prob_val[:, 1]
    y_val = pred_val.label_ids

    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 19):
        y_pred = (prob_val >= t).astype(np.int64)
        m = evaluate_binary(y_val, y_pred, prob_val)
        if m["f1"] >= best_f1:
            best_f1 = m["f1"]
            best_t = float(t)

    # Save threshold for evaluation
    with open(out_dir / "results.json", "w") as f:
        json.dump({"decision_threshold": best_t}, f)
    return {"threshold": best_t, "val_metrics": evaluate_binary(y_val, (prob_val >= best_t).astype(np.int64), prob_val)}


def train_fusion(
    model_id: str,
    tr_pairs: list,
    val_pairs: list,
    test_pairs: list,
    acoustic_csv: Path,
    ckpt_dir: Path,
    device,
) -> dict:
    """Train fusion model. Acoustic features from noisy csv; scaler fit on noisy train only."""
    short = model_short_name(model_id)
    df = pd.read_csv(acoustic_csv)
    if df.empty:
        raise RuntimeError(f"Empty acoustic CSV: {acoustic_csv}")

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
            key = str(Path(p).resolve())
            if key not in path_to_idx and Path(p).resolve().as_posix() not in path_to_idx:
                continue
            xs.append(p)
            ys.append(l)
        return xs, ys

    tr_paths, tr_labels = subset(tr_pairs)
    val_paths, val_labels = subset(val_pairs)
    te_paths, te_labels = subset(test_pairs)

    def matrix(paths):
        rows = []
        for p in paths:
            key = str(Path(p).resolve())
            if key not in path_to_idx:
                key = Path(p).resolve().as_posix()
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
    num_steps = MAX_EPOCHS * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * num_steps), num_steps)

    best_val_auc = 0.0
    patience_counter = 0
    ckpt_path = ckpt_dir / f"fusion_{short}.pt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"{short} epoch {epoch}", leave=False):
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

        model.eval()
        all_prob, all_y = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    input_values=batch["input_values"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    acoustic=batch["acoustic"].to(device),
                )
                prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_prob.extend(prob)
                all_y.extend(batch["labels"].numpy())
        all_prob = np.array(all_prob)
        all_y = np.array(all_y)
        val_metrics = evaluate_binary(all_y, (all_prob >= 0.5).astype(np.int64), all_prob)
        val_auc = val_metrics["auc_roc"]

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
        if patience_counter >= 5:
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
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

    prob_te, y_te = collect_probs(test_loader)
    test_metrics = evaluate_binary(y_te, (prob_te >= best_t).astype(np.int64), prob_te)
    test_metrics["decision_threshold"] = best_t

    # Save scaler and threshold for evaluation (shared across models in same condition)
    joblib.dump(scaler, ckpt_dir / "scaler.joblib")
    with open(ckpt_dir / f"threshold_{short}.json", "w") as f:
        json.dump({"threshold": best_t}, f)
    return test_metrics


def evaluate_base_transformer(
    model_dir: Path,
    test_paths: list,
    test_labels: list,
    device,
) -> dict:
    """Evaluate fine-tuned base transformer on test set."""
    model = AutoModelForAudioClassification.from_pretrained(str(model_dir))
    fe = AutoFeatureExtractor.from_pretrained(str(model_dir))
    model = model.to(device).eval()

    max_samples = int(5.0 * SAMPLE_RATE)
    probs = []
    for i in range(0, len(test_paths), 16):
        batch_paths = test_paths[i : i + 16]
        arrays = []
        for p in batch_paths:
            y, _ = librosa.load(str(p), sr=SAMPLE_RATE, mono=True)
            if len(y) > max_samples:
                start = (len(y) - max_samples) // 2
                y = y[start : start + max_samples]
            elif len(y) < max_samples:
                y = np.pad(y, (0, max_samples - len(y)), mode="constant")
            arrays.append(y.astype(np.float32))
        inputs = fe(
            arrays, sampling_rate=SAMPLE_RATE, return_tensors="pt",
            padding=True, truncation=True, max_length=max_samples,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

    probs = np.array(probs)
    # Load threshold from results.json if present
    res_file = model_dir / "results.json"
    thresh = 0.5
    if res_file.exists():
        try:
            with open(res_file) as f:
                thresh = json.load(f).get("decision_threshold", 0.5)
        except Exception:
            pass
    preds = (probs >= thresh).astype(np.int64)
    return evaluate_binary(np.array(test_labels), preds, probs)


def evaluate_fusion_model(
    model_id: str,
    ckpt_path: Path,
    test_pairs: list,
    acoustic_csv: Path,
    device,
) -> dict:
    """Evaluate fusion model on test set. Threshold from validation (stored in training)."""
    short = model_short_name(model_id)
    df = pd.read_csv(acoustic_csv)
    feat_cols = [c for c in df.columns if c not in ("path", "label")]

    scaler_path = ckpt_path.parent / "scaler.joblib"
    if not scaler_path.exists():
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auc_roc": 0, "confusion_matrix": [[0, 0], [0, 0]]}
    scaler = joblib.load(scaler_path)

    test_paths = [p for p, _ in test_pairs]
    test_labels = [l for _, l in test_pairs]
    test_ds = FusionDataset(test_paths, test_labels, df, scaler, feat_cols,
                           AutoFeatureExtractor.from_pretrained(model_id))
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = TransformerFusionModel(model_id, acoustic_dim=len(feat_cols)).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_values=batch["input_values"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                acoustic=batch["acoustic"].to(device),
            )
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            labels.extend(batch["labels"].numpy())

    probs = np.array(probs)
    labels = np.array(labels)
    thresh_path = ckpt_path.parent / f"threshold_{short}.json"
    thresh = 0.5
    if thresh_path.exists():
        try:
            thresh = json.load(open(thresh_path)).get("threshold", 0.5)
        except Exception:
            pass
    preds = (probs >= thresh).astype(np.int64)
    return evaluate_binary(labels, preds, probs)


def main() -> int:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    split_path = RESULTS_DIR / SPLIT_FILENAME
    loaded = load_split(split_path)
    if loaded is None:
        print("results/split.json missing. Run: python scripts/train_cnn.py first.")
        return 1

    tr_pairs, val_pairs, test_pairs = loaded
    print(f"Split: train={len(tr_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")

    # --- Part 1: Create noisy datasets ---
    print("\n=== Part 1: Noisy datasets ===")
    all_meta = ensure_noisy_datasets(tr_pairs, val_pairs, test_pairs)

    # --- Part 2: Extract acoustic features for each noisy condition ---
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    for key in all_meta:
        csv_path = FEATURES_DIR / f"acoustic_noisy_{key}.csv"
        if not csv_path.exists():
            root = Path(all_meta[key]["root"])
            extract_features_from_dir(root, csv_path)

    # --- Part 3 & 4: Train and evaluate ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load clean baseline from Stage 6 and Stage 7 results (no re-eval to avoid scaler issues)
    clean_base = {}
    clean_fusion = {}
    tr_results = OUTPUTS_DIR / "transformer_results.json"
    fusion_results = OUTPUTS_DIR / "fusion_models_results.json"
    for short in ["hubert_base_ls960", "wavlm"]:
        clean_base[short] = None
        clean_fusion[short] = None
    if tr_results.exists():
        with open(tr_results) as f:
            tr_data = json.load(f)
        for short in ["hubert_base_ls960", "wavlm"]:
            if short in tr_data:
                m = tr_data[short]
                clean_base[short] = {k: v for k, v in m.items() if k in ("accuracy", "precision", "recall", "f1", "auc_roc")}
    if fusion_results.exists():
        with open(fusion_results) as f:
            fusion_data = json.load(f)
        for short in ["hubert_base_ls960", "wavlm"]:
            if short in fusion_data:
                m = fusion_data[short]
                clean_fusion[short] = {k: v for k, v in m.items() if k in ("accuracy", "precision", "recall", "f1", "auc_roc")}

    # Results structure: {model: {condition: metrics}}
    results = {
        "clean": {"base": clean_base, "fusion": clean_fusion},
        "noisy": {},
        "thresholds": {},
    }

    for key in sorted(all_meta.keys()):
        print(f"\n=== Condition: {key} ===")
        meta = all_meta[key]
        root = Path(meta["root"])
        csv_path = FEATURES_DIR / f"acoustic_noisy_{key}.csv"

        tr_noisy = pairs_from_metadata(meta, tr_pairs, "train")
        val_noisy = pairs_from_metadata(meta, val_pairs, "val")
        te_noisy = pairs_from_metadata(meta, test_pairs, "test")

        tr_paths = [str(p) for p, _ in tr_noisy]
        tr_labels = [l for _, l in tr_noisy]
        val_paths = [str(p) for p, _ in val_noisy]
        val_labels = [l for _, l in val_noisy]
        te_paths = [str(p) for p, _ in te_noisy]
        te_labels = [l for _, l in te_noisy]

        results["noisy"][key] = {"base": {}, "fusion": {}}

        # Train base transformers
        for model_id in BASE_MODELS:
            short = model_short_name(model_id)
            out_dir = OUTPUTS_DIR / "noisy" / key / f"transformer_{short}"
            if not (out_dir / "config.json").exists():
                print(f"  Training base {short}...")
                train_base_transformer(
                    model_id, tr_paths, tr_labels, val_paths, val_labels, out_dir, device
                )
            else:
                print(f"  Base {short} already trained")

            metrics = evaluate_base_transformer(out_dir, te_paths, te_labels, device)
            results["noisy"][key]["base"][short] = metrics

        # Train fusion models
        for model_id in BASE_MODELS:
            short = model_short_name(model_id)
            ckpt_dir = OUTPUTS_DIR / "noisy" / key / "fusion"
            ckpt_path = ckpt_dir / f"fusion_{short}.pt"
            if not ckpt_path.exists():
                print(f"  Training fusion {short}...")
                m = train_fusion(model_id, tr_noisy, val_noisy, te_noisy, csv_path, ckpt_dir, device)
                results["noisy"][key]["fusion"][short] = m
            else:
                metrics = evaluate_fusion_model(model_id, ckpt_path, te_noisy, csv_path, device)
                results["noisy"][key]["fusion"][short] = metrics

    # --- Part 5 & 6: Comparison tables and plots ---
    # Build comparison DataFrame
    snr_labels = ["clean"] + [f"{int(s)}dB" for s in SNR_LEVELS_NOISY]

    # For each noise type, build table
    tables = {}
    for noise_type in NOISE_TYPES:
        rows = []
        for model_name in ["hubert_base_ls960", "wavlm"]:
            base_row = {"Model": f"{model_name} (base)", "Clean AUC": "-", "20dB": "-", "10dB": "-", "5dB": "-", "0dB": "-"}
            fusion_row = {"Model": f"{model_name} (fusion)", "Clean AUC": "-", "20dB": "-", "10dB": "-", "5dB": "-", "0dB": "-"}

            if results["clean"]["base"].get(model_name) and results["clean"]["base"][model_name]:
                base_row["Clean AUC"] = f"{results['clean']['base'][model_name]['auc_roc']:.3f}"
            if results["clean"]["fusion"].get(model_name) and results["clean"]["fusion"][model_name]:
                fusion_row["Clean AUC"] = f"{results['clean']['fusion'][model_name]['auc_roc']:.3f}"

            for snr in SNR_LEVELS_NOISY:
                key = f"{noise_type}_{int(snr)}dB"
                if key in results["noisy"]:
                    b = results["noisy"][key]["base"].get(model_name, {})
                    f = results["noisy"][key]["fusion"].get(model_name, {})
                    base_row[f"{int(snr)}dB"] = f"{b.get('auc_roc', 0):.3f}" if b else "-"
                    fusion_row[f"{int(snr)}dB"] = f"{f.get('auc_roc', 0):.3f}" if f else "-"

            rows.append(base_row)
            rows.append(fusion_row)
        tables[noise_type] = pd.DataFrame(rows)

    # --- Part 6: Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax_idx, noise_type in enumerate(NOISE_TYPES):
        ax_auc = axes[0, ax_idx] if len(NOISE_TYPES) > 1 else axes[ax_idx]
        ax_f1 = axes[1, ax_idx] if len(NOISE_TYPES) > 1 else axes[ax_idx + 2]

        x = ["clean"] + [f"{s}dB" for s in SNR_LEVELS_NOISY]
        for model_name in ["hubert_base_ls960", "wavlm"]:
            auc_vals = []
            f1_vals = []
            m = results["clean"]["base"].get(model_name)
            if m and isinstance(m, dict):
                auc_vals.append(m.get("auc_roc", np.nan))
                f1_vals.append(m.get("f1", np.nan))
            else:
                auc_vals.append(np.nan)
                f1_vals.append(np.nan)
            for snr in SNR_LEVELS_NOISY:
                key = f"{noise_type}_{int(snr)}dB"
                m = results["noisy"].get(key, {}).get("base", {}).get(model_name, {})
                auc_vals.append(m.get("auc_roc", np.nan))
                f1_vals.append(m.get("f1", np.nan))

            ax_auc.plot(x, auc_vals, "-o", label=f"{model_name} base")
            ax_f1.plot(x, f1_vals, "-o", label=f"{model_name} base")

            mf = results["clean"]["fusion"].get(model_name)
            auc_vals = [mf.get("auc_roc", np.nan) if mf else np.nan]
            f1_vals = [mf.get("f1", np.nan) if mf else np.nan]
            for snr in SNR_LEVELS_NOISY:
                key = f"{noise_type}_{int(snr)}dB"
                m = results["noisy"].get(key, {}).get("fusion", {}).get(model_name, {})
                auc_vals.append(m.get("auc_roc", np.nan))
                f1_vals.append(m.get("f1", np.nan))
            ax_auc.plot(x, auc_vals, "--s", label=f"{model_name} fusion")
            ax_f1.plot(x, f1_vals, "--s", label=f"{model_name} fusion")

        ax_auc.set_title(f"{noise_type} - AUC vs SNR")
        ax_auc.set_ylim(0, 1.05)
        ax_auc.legend()
        ax_f1.set_title(f"{noise_type} - F1 vs SNR")
        ax_f1.set_ylim(0, 1.05)
        ax_f1.legend()

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "noise_robustness_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Part 7: Analysis ---
    analysis = []
    for noise_type in NOISE_TYPES:
        for snr in SNR_LEVELS_NOISY:
            key = f"{noise_type}_{int(snr)}dB"
            if key not in results["noisy"]:
                continue
            base_aucs = [results["noisy"][key]["base"].get(m, {}).get("auc_roc", 0) for m in ["hubert_base_ls960", "wavlm"]]
            fusion_aucs = [results["noisy"][key]["fusion"].get(m, {}).get("auc_roc", 0) for m in ["hubert_base_ls960", "wavlm"]]
            avg_base = np.nanmean(base_aucs) if base_aucs else 0
            avg_fusion = np.nanmean(fusion_aucs) if fusion_aucs else 0
            if avg_fusion > avg_base + 0.02:
                analysis.append(f"{key}: Fusion improves robustness (fusion={avg_fusion:.2f} vs base={avg_base:.2f})")
            elif avg_base > avg_fusion + 0.02:
                analysis.append(f"{key}: Base more robust (base={avg_base:.2f} vs fusion={avg_fusion:.2f})")
            else:
                analysis.append(f"{key}: Similar performance (base={avg_base:.2f}, fusion={avg_fusion:.2f})")

    # --- Part 8: Save results ---
    def _serialize_metrics(m):
        if m is None:
            return None
        return {kk: vv for kk, vv in m.items()}

    out = {
        "clean": {
            "base": {k: _serialize_metrics(v) for k, v in results["clean"]["base"].items()},
            "fusion": {k: _serialize_metrics(v) for k, v in results["clean"]["fusion"].items()},
        },
        "noisy": {
            k: {
                "base": {m: _serialize_metrics(v) for m, v in data["base"].items()},
                "fusion": {m: _serialize_metrics(v) for m, v in data["fusion"].items()},
            }
            for k, data in results["noisy"].items()
        },
        "tables": {k: v.to_dict(orient="records") for k, v in tables.items()},
        "analysis": analysis,
    }

    out_path = RESULTS_DIR / "noise_robustness.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved to {out_path}")
    print("\n=== Analysis ===")
    for line in analysis:
        print(f"  {line}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
