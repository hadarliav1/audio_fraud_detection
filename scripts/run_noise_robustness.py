#!/usr/bin/env python3
"""
Stage 8 — Noise Robustness: Measure model degradation under white, pink, and compression noise.

For each model (transformers, CNN, RF, hubert_frozen_lr, fusion) and each (noise_type, SNR),
corrupt test audio and evaluate. Output: degradation curves for analysis.

- hubert_base_ls960 (etc.): fine-tuned transformer from checkpoint (encoder + trained head).
- hubert_frozen_lr: frozen HuBERT encoder + LR on embeddings only — same pipeline as notebook 07
  "HuBERT Embeddings Only", for fair comparison with Fusion.
- Fusion: frozen HuBERT embeddings + acoustic features + LR (trained on clean, evaluated on noisy).

Noise types: white (flat spectrum), pink (1/f), compression (codec-like simulation).
SNR levels: clean (inf), 20, 10, 5, 0 dB.
"""

import json
import sys
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import torch
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    F0_FMAX,
    F0_FMIN,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    NOISE_TYPES,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    SAMPLE_RATE,
    SNR_LEVELS_DB,
    TEST_SIZE,
    TRANSFORMER_MODELS,
    VAL_SIZE,
)
from src.features.acoustic_scipy import extract_all_features, features_to_vector, get_feature_names
from src.models.cnn_spectrogram import SpectrogramCNN
from src.utils.audio import load_audio
from src.utils.eval import evaluate_binary
from src.utils.paths import get_audio_paths_with_labels
from src.utils.splits import speaker_disjoint_split

MAX_AUDIO_SAMPLES = int(5.0 * SAMPLE_RATE)
BEST_TRANSFORMER = "facebook/hubert-base-ls960"
FUSION_TOP_ACOUSTIC_N = 30
FUSION_CORR_THRESHOLD = 0.85
LR_C = 0.1


def model_short_name(model_id: str) -> str:
    """Match train_transformers naming exactly for checkpoint dirs."""
    name = model_id.split("/")[-1].split(".")[0]
    for suffix in ["-base", "-tiny", "-small", "_base", "_tiny", "_small"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace("-", "_")


def _get_encoder(model):
    """Get encoder submodule (hubert, wav2vec2, wavlm)."""
    for attr in ("hubert", "wav2vec2", "wavlm"):
        enc = getattr(model, attr, None)
        if enc is not None:
            return enc
    raise ValueError("Unknown model type: no encoder found")


def extract_embeddings_from_arrays(model, feature_extractor, arrays, device, batch_size=16):
    """Extract mean-pooled embeddings from audio arrays."""
    model.eval()
    encoder = _get_encoder(model)
    embeddings = []
    for i in range(0, len(arrays), batch_size):
        batch = arrays[i : i + batch_size]
        batch = [a.astype(np.float32) for a in batch]
        inputs = feature_extractor(
            batch, sampling_rate=SAMPLE_RATE, return_tensors="pt",
            padding=True, truncation=True, max_length=MAX_AUDIO_SAMPLES,
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


def select_acoustic_features(X: np.ndarray, y: np.ndarray, top_n: int = FUSION_TOP_ACOUSTIC_N, corr_threshold: float = FUSION_CORR_THRESHOLD) -> np.ndarray:
    """Select acoustic features: top by univariate AUC, then remove highly correlated."""
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


def add_white_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise to achieve target SNR (dB)."""
    if snr_db == float("inf"):
        return y.copy()
    sig_power = np.mean(y ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.standard_normal(len(y), dtype=np.float32) * np.sqrt(noise_power)
    return y + noise


def add_pink_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add pink (1/f) noise. Generate via filtering white noise."""
    if snr_db == float("inf"):
        return y.copy()
    white = rng.standard_normal(len(y), dtype=np.float32)
    # Approximate 1/f: integrate white -> brown, average with white for pink
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    pink = signal.lfilter(b, a, white)
    pink = pink / (np.std(pink) + 1e-8)
    sig_power = np.mean(y ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    return y + pink * np.sqrt(noise_power)


def add_compression_noise(y: np.ndarray, snr_db: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate codec artifacts: bandwidth limit + quantization. SNR controls noise level."""
    if snr_db == float("inf"):
        return y.copy()
    # Downsample to 8kHz and back (telephony-like)
    y_8k = librosa.resample(y.astype(np.float64), orig_sr=sr, target_sr=8000)
    y_back = librosa.resample(y_8k, orig_sr=8000, target_sr=sr)
    if len(y_back) > len(y):
        y_back = y_back[: len(y)]
    elif len(y_back) < len(y):
        y_back = np.pad(y_back, (0, len(y) - len(y_back)), mode="edge")
    # Add quantization-like noise
    n_bits = max(4, int(16 - (0 - snr_db) / 6))
    scale = 2 ** (n_bits - 1)
    quantized = np.round(y_back * scale) / scale
    # Blend with original based on SNR
    alpha = 1.0 / (1.0 + 10 ** (-snr_db / 10))
    return (1 - alpha) * y + alpha * quantized


def corrupt_audio(y: np.ndarray, noise_type: str, snr_db: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    if noise_type == "white":
        return add_white_noise(y, snr_db, rng)
    if noise_type == "pink":
        return add_pink_noise(y, snr_db, rng)
    if noise_type == "compression":
        return add_compression_noise(y, snr_db, sr, rng)
    return y.copy()


def evaluate_transformers(test_paths, test_labels, noise_type, snr_db, model, feature_extractor, device, rng, max_samples=None):
    """Run transformer on corrupted audio."""
    model.eval()
    preds, probs = [], []
    max_samples = max_samples or MAX_AUDIO_SAMPLES
    for i in range(0, len(test_paths), 16):
        batch_paths = test_paths[i : i + 16]
        arrays = []
        for p in batch_paths:
            y = load_audio(p, sr=SAMPLE_RATE)
            y = corrupt_audio(y.astype(np.float32), noise_type, snr_db, SAMPLE_RATE, rng)
            if len(y) > max_samples:
                start = (len(y) - max_samples) // 2
                y = y[start : start + max_samples]
            elif len(y) < max_samples:
                y = np.pad(y, (0, max_samples - len(y)), mode="constant")
            arrays.append(y)
        inputs = feature_extractor(
            arrays, sampling_rate=SAMPLE_RATE, return_tensors="pt",
            padding=True, truncation=True, max_length=max_samples,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits
        probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    return evaluate_binary(np.array(test_labels), np.array(preds), np.array(probs))


def evaluate_cnn(test_paths, test_labels, noise_type, snr_db, model, device, rng):
    """Run CNN on mel-spectrogram from corrupted audio."""
    model.eval()
    preds, probs = [], []
    max_frames = int(5.0 * SAMPLE_RATE / HOP_LENGTH)
    for p, lb in zip(test_paths, test_labels):
        y = load_audio(p, sr=SAMPLE_RATE)
        y = corrupt_audio(y.astype(np.float32), noise_type, snr_db, SAMPLE_RATE, rng)
        mel = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        T = mel_db.shape[1]
        if T > max_frames:
            start = (T - max_frames) // 2
            mel_db = mel_db[:, start : start + max_frames]
        elif T < max_frames:
            mel_db = np.pad(mel_db, ((0, 0), (0, max_frames - T)), mode="constant")
        x = torch.from_numpy(mel_db).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
        probs.append(torch.softmax(logits, dim=1)[0, 1].item())
        preds.append(torch.argmax(logits, dim=1)[0].item())
    return evaluate_binary(np.array(test_labels), np.array(preds), np.array(probs))


def evaluate_rf(test_paths, test_labels, noise_type, snr_db, pipe, rng):
    """Extract features from corrupted audio, run RF."""
    feat_names = get_feature_names()
    X = []
    for p in test_paths:
        y = load_audio(p, sr=SAMPLE_RATE)
        y = corrupt_audio(y.astype(np.float32), noise_type, snr_db, SAMPLE_RATE, rng)
        feat = extract_all_features(
            y, sr=SAMPLE_RATE, n_mfcc=13, n_mels=N_MELS, hop_length=HOP_LENGTH,
            n_fft=N_FFT, f0_fmin=F0_FMIN, f0_fmax=F0_FMAX,
        )
        vec = features_to_vector(feat)
        X.append(vec)
    X = np.array(X)
    df = pd.DataFrame(X, columns=feat_names)
    df = df.fillna(df.median())
    pred = pipe.predict(df)
    prob = pipe.predict_proba(df)[:, 1]
    return evaluate_binary(np.array(test_labels), pred, prob)


def evaluate_frozen_hubert_lr(test_paths, test_labels, noise_type, snr_db, state, device, rng):
    """Run frozen HuBERT encoder + LR (no acoustic). Same pipeline as 07 'HuBERT Embeddings Only'."""
    scaler, lr, model, feature_extractor = (
        state["scaler"], state["lr"], state["model"], state["feature_extractor"],
    )
    arrays = []
    for p in test_paths:
        y = load_audio(p, sr=SAMPLE_RATE)
        y = corrupt_audio(y.astype(np.float32), noise_type, snr_db, SAMPLE_RATE, rng)
        if len(y) > MAX_AUDIO_SAMPLES:
            start = (len(y) - MAX_AUDIO_SAMPLES) // 2
            y = y[start : start + MAX_AUDIO_SAMPLES]
        elif len(y) < MAX_AUDIO_SAMPLES:
            y = np.pad(y, (0, MAX_AUDIO_SAMPLES - len(y)), mode="constant")
        arrays.append(y)
    emb = extract_embeddings_from_arrays(model, feature_extractor, arrays, device)
    X_s = scaler.transform(emb)
    pred = lr.predict(X_s)
    prob = lr.predict_proba(X_s)[:, 1]
    return evaluate_binary(np.array(test_labels), pred, prob)


def evaluate_fusion(test_paths, test_labels, noise_type, snr_db, fusion_state, device, rng):
    """Run fusion (HuBERT emb + acoustic) on corrupted audio."""
    scaler, lr, sel_idx, model, feature_extractor = (
        fusion_state["scaler"], fusion_state["lr"], fusion_state["sel_idx"],
        fusion_state["model"], fusion_state["feature_extractor"],
    )
    arrays = []
    X_ac = []
    for p in test_paths:
        y = load_audio(p, sr=SAMPLE_RATE)
        y = corrupt_audio(y.astype(np.float32), noise_type, snr_db, SAMPLE_RATE, rng)
        if len(y) > MAX_AUDIO_SAMPLES:
            start = (len(y) - MAX_AUDIO_SAMPLES) // 2
            y = y[start : start + MAX_AUDIO_SAMPLES]
        elif len(y) < MAX_AUDIO_SAMPLES:
            y = np.pad(y, (0, MAX_AUDIO_SAMPLES - len(y)), mode="constant")
        arrays.append(y)
        feat = extract_all_features(
            y, sr=SAMPLE_RATE, n_mfcc=13, n_mels=N_MELS, hop_length=HOP_LENGTH,
            n_fft=N_FFT, f0_fmin=F0_FMIN, f0_fmax=F0_FMAX,
        )
        vec = features_to_vector(feat)
        X_ac.append(vec)
    X_ac = np.array(X_ac)
    X_ac = np.nan_to_num(X_ac, nan=np.nanmedian(X_ac, axis=0))
    X_ac = X_ac[:, sel_idx]
    emb = extract_embeddings_from_arrays(model, feature_extractor, arrays, device)
    X = np.hstack([emb, X_ac])
    X_s = scaler.transform(X)
    pred = lr.predict(X_s)
    prob = lr.predict_proba(X_s)[:, 1]
    return evaluate_binary(np.array(test_labels), pred, prob)


def main() -> int:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = get_audio_paths_with_labels(PROCESSED_DIR)
    if not pairs:
        print("No processed audio. Run: python scripts/run_preprocessing.py")
        return 1

    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    tr_pairs, _, test_pairs = speaker_disjoint_split(pairs, TEST_SIZE, val_ratio, RANDOM_SEED)
    train_paths = [p for p, _ in tr_pairs]
    train_labels = np.array([l for _, l in tr_pairs])
    test_paths = [p for p, _ in test_pairs]
    test_labels = [l for _, l in test_pairs]
    print(f"Train set: {len(train_paths)}, Test set: {len(test_paths)} samples")

    results = {}

    # 1. Transformers (all with checkpoints: HuBERT, Wav2Vec2, WavLM, Whisper)
    for model_id in TRANSFORMER_MODELS:
        short = model_short_name(model_id)
        ckpt_dir = OUTPUTS_DIR / f"transformer_{short}"
        if not ckpt_dir.exists():
            print(f"Skipping {short} (no checkpoint at {ckpt_dir})")
            continue
        print(f"\nEvaluating {short}...")
        model = AutoModelForAudioClassification.from_pretrained(str(ckpt_dir))
        feature_extractor = AutoFeatureExtractor.from_pretrained(str(ckpt_dir))
        model = model.to(device)
        # Whisper expects 30s; others use 5s
        max_samples = int(30 * SAMPLE_RATE) if "whisper" in model_id.lower() else MAX_AUDIO_SAMPLES
        results[short] = {}
        for noise_type in NOISE_TYPES:
            results[short][noise_type] = {}
            for snr in SNR_LEVELS_DB:
                label = "clean" if snr == float("inf") else f"{int(snr)}dB"
                m = evaluate_transformers(
                    test_paths, test_labels, noise_type, snr,
                    model, feature_extractor, device, rng, max_samples=max_samples
                )
                results[short][noise_type][label] = {"auc": m["auc_roc"], "f1": m["f1"]}
                print(f"  {noise_type} {label}: AUC={m['auc_roc']:.3f}")
    if not any(r.startswith("hubert") or r.startswith("wav2vec") or r.startswith("wavlm") or r.startswith("whisper") for r in results):
        print("No transformer checkpoints found. Run: python scripts/train_transformers.py")

    # 2. CNN
    cnn_path = OUTPUTS_DIR / "cnn_spectrogram.pt"
    if cnn_path.exists():
        print("\nEvaluating CNN...")
        model = SpectrogramCNN(n_mels=N_MELS, n_classes=2)
        model.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
        results["cnn"] = {}
        for noise_type in NOISE_TYPES:
            results["cnn"][noise_type] = {}
            for snr in SNR_LEVELS_DB:
                label = "clean" if snr == float("inf") else f"{int(snr)}dB"
                m = evaluate_cnn(test_paths, test_labels, noise_type, snr, model, device, rng)
                results["cnn"][noise_type][label] = {"auc": m["auc_roc"], "f1": m["f1"]}
                print(f"  {noise_type} {label}: AUC={m['auc_roc']:.3f}")
    else:
        print("No CNN checkpoint. Run: python scripts/train_cnn.py")

    # 3. RF (acoustic baseline)
    rf_path = OUTPUTS_DIR / "baseline_random_forest.joblib"
    if rf_path.exists():
        print("\nEvaluating RF...")
        pipe = joblib.load(rf_path)
        results["rf"] = {}
        for noise_type in NOISE_TYPES:
            results["rf"][noise_type] = {}
            for snr in SNR_LEVELS_DB:
                label = "clean" if snr == float("inf") else f"{int(snr)}dB"
                m = evaluate_rf(test_paths, test_labels, noise_type, snr, pipe, rng)
                results["rf"][noise_type][label] = {"auc": m["auc_roc"], "f1": m["f1"]}
                print(f"  {noise_type} {label}: AUC={m['auc_roc']:.3f}")
    else:
        print("No RF checkpoint. Run: python scripts/train_baseline.py")

    # 4a. HuBERT frozen + LR (same as 07 "HuBERT Embeddings Only" — fair comparison to Fusion)
    print("\nEvaluating HuBERT (frozen + LR, same as notebook 07)...")
    model_frozen = AutoModelForAudioClassification.from_pretrained(BEST_TRANSFORMER, num_labels=2)
    feat_ext_frozen = AutoFeatureExtractor.from_pretrained(BEST_TRANSFORMER)
    model_frozen = model_frozen.to(device)
    train_arrays_emb = []
    for p in train_paths:
        y = load_audio(p, sr=SAMPLE_RATE)
        if len(y) > MAX_AUDIO_SAMPLES:
            start = (len(y) - MAX_AUDIO_SAMPLES) // 2
            y = y[start : start + MAX_AUDIO_SAMPLES]
        elif len(y) < MAX_AUDIO_SAMPLES:
            y = np.pad(y, (0, MAX_AUDIO_SAMPLES - len(y)), mode="constant")
        train_arrays_emb.append(y.astype(np.float32))
    tr_emb_only = extract_embeddings_from_arrays(model_frozen, feat_ext_frozen, train_arrays_emb, device)
    scaler_emb = StandardScaler()
    X_tr_emb_s = scaler_emb.fit_transform(tr_emb_only)
    lr_emb = LogisticRegression(C=LR_C, max_iter=1000, random_state=RANDOM_SEED)
    lr_emb.fit(X_tr_emb_s, train_labels)
    hubert_frozen_state = {"scaler": scaler_emb, "lr": lr_emb, "model": model_frozen, "feature_extractor": feat_ext_frozen}
    results["hubert_frozen_lr"] = {}
    for noise_type in NOISE_TYPES:
        results["hubert_frozen_lr"][noise_type] = {}
        for snr in SNR_LEVELS_DB:
            label = "clean" if snr == float("inf") else f"{int(snr)}dB"
            m = evaluate_frozen_hubert_lr(test_paths, test_labels, noise_type, snr, hubert_frozen_state, device, rng)
            results["hubert_frozen_lr"][noise_type][label] = {"auc": m["auc_roc"], "f1": m["f1"]}
            print(f"  {noise_type} {label}: AUC={m['auc_roc']:.3f}")

    # 4b. Fusion (frozen HuBERT + acoustic features + LR)
    print("\nEvaluating Fusion (HuBERT + Acoustic)...")
    model = AutoModelForAudioClassification.from_pretrained(BEST_TRANSFORMER, num_labels=2)
    feature_extractor = AutoFeatureExtractor.from_pretrained(BEST_TRANSFORMER)
    model = model.to(device)
    # Extract embeddings from clean train
    train_arrays = []
    for p in train_paths:
        y = load_audio(p, sr=SAMPLE_RATE)
        if len(y) > MAX_AUDIO_SAMPLES:
            start = (len(y) - MAX_AUDIO_SAMPLES) // 2
            y = y[start : start + MAX_AUDIO_SAMPLES]
        elif len(y) < MAX_AUDIO_SAMPLES:
            y = np.pad(y, (0, MAX_AUDIO_SAMPLES - len(y)), mode="constant")
        train_arrays.append(y.astype(np.float32))
    tr_emb = extract_embeddings_from_arrays(model, feature_extractor, train_arrays, device)
    # Extract acoustic from clean train
    X_tr_ac = []
    for p in train_paths:
        y = load_audio(p, sr=SAMPLE_RATE)
        feat = extract_all_features(
            y, sr=SAMPLE_RATE, n_mfcc=13, n_mels=N_MELS, hop_length=HOP_LENGTH,
            n_fft=N_FFT, f0_fmin=F0_FMIN, f0_fmax=F0_FMAX,
        )
        X_tr_ac.append(features_to_vector(feat))
    X_tr_ac = np.array(X_tr_ac)
    X_tr_ac = np.nan_to_num(X_tr_ac, nan=np.nanmedian(X_tr_ac, axis=0))
    sel_idx = select_acoustic_features(X_tr_ac, train_labels, top_n=FUSION_TOP_ACOUSTIC_N, corr_threshold=FUSION_CORR_THRESHOLD)
    X_tr_ac = X_tr_ac[:, sel_idx]
    X_tr = np.hstack([tr_emb, X_tr_ac])
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    lr = LogisticRegression(C=LR_C, max_iter=1000, random_state=RANDOM_SEED)
    lr.fit(X_tr_s, train_labels)
    fusion_state = {"scaler": scaler, "lr": lr, "sel_idx": sel_idx, "model": model, "feature_extractor": feature_extractor}
    results["fusion"] = {}
    for noise_type in NOISE_TYPES:
        results["fusion"][noise_type] = {}
        for snr in SNR_LEVELS_DB:
            label = "clean" if snr == float("inf") else f"{int(snr)}dB"
            m = evaluate_fusion(test_paths, test_labels, noise_type, snr, fusion_state, device, rng)
            results["fusion"][noise_type][label] = {"auc": m["auc_roc"], "f1": m["f1"]}
            print(f"  {noise_type} {label}: AUC={m['auc_roc']:.3f}")

    with open(OUTPUTS_DIR / "noise_robustness_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUTS_DIR / 'noise_robustness_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
