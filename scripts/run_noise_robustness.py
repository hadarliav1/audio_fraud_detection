#!/usr/bin/env python3
"""
Stage 8 — Noise Robustness: Measure model degradation under white, pink, and compression noise.

For each model (transformers, CNN, RF) and each (noise_type, SNR), corrupt test audio
and evaluate. Output: degradation curves for analysis.

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


def model_short_name(model_id: str) -> str:
    """Match train_transformers naming exactly for checkpoint dirs."""
    name = model_id.split("/")[-1].split(".")[0]
    for suffix in ["-base", "-tiny", "-small", "_base", "_tiny", "_small"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace("-", "_")


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
    _, _, test_pairs = speaker_disjoint_split(pairs, TEST_SIZE, val_ratio, RANDOM_SEED)
    test_paths = [p for p, _ in test_pairs]
    test_labels = [l for _, l in test_pairs]
    print(f"Test set: {len(test_paths)} samples")

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

    with open(OUTPUTS_DIR / "noise_robustness_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUTS_DIR / 'noise_robustness_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
