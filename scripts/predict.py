#!/usr/bin/env python3
"""
Predict real vs fake for a single audio file.

  python scripts/predict.py path/to/audio.wav
  python scripts/predict.py --model {wav2vec2|cnn|rf} path/to/audio.wav
"""

import argparse
import sys
from pathlib import Path

import joblib
import librosa
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    F0_FMAX,
    F0_FMIN,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    OUTPUTS_DIR,
    SAMPLE_RATE,
)
from src.features.acoustic_scipy import extract_all_features, features_to_vector, get_feature_names
from src.models.cnn_spectrogram import SpectrogramCNN
from src.utils.audio import load_audio

MAX_AUDIO_SAMPLES = int(5.0 * SAMPLE_RATE)
MAX_FRAMES = int(5.0 * SAMPLE_RATE / HOP_LENGTH)


def predict_rf(audio_path: Path) -> tuple:
    """RF: acoustic features -> predict."""
    pipe = joblib.load(OUTPUTS_DIR / "baseline_random_forest.joblib")
    y = load_audio(audio_path, sr=SAMPLE_RATE)
    feat = extract_all_features(
        y, sr=SAMPLE_RATE, n_mfcc=13, n_mels=N_MELS, hop_length=HOP_LENGTH,
        n_fft=N_FFT, f0_fmin=F0_FMIN, f0_fmax=F0_FMAX,
    )
    vec = features_to_vector(feat)
    X = np.array([vec])
    df = __import__("pandas").DataFrame(X, columns=get_feature_names())
    df = df.fillna(0)
    prob = pipe.predict_proba(df)[0, 1]
    pred = 1 if prob >= 0.5 else 0
    return pred, prob


def predict_cnn(audio_path: Path, device: torch.device) -> tuple:
    """CNN: mel-spectrogram -> predict."""
    model = SpectrogramCNN(n_mels=N_MELS, n_classes=2)
    model.load_state_dict(torch.load(OUTPUTS_DIR / "cnn_spectrogram.pt", map_location=device, weights_only=True))
    model = model.to(device).eval()

    y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    T = mel_db.shape[1]
    if T > MAX_FRAMES:
        start = (T - MAX_FRAMES) // 2
        mel_db = mel_db[:, start : start + MAX_FRAMES]
    elif T < MAX_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, MAX_FRAMES - T)), mode="constant")

    x = torch.from_numpy(mel_db).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    prob = torch.softmax(logits, dim=1)[0, 1].item()
    pred = 1 if prob >= 0.5 else 0
    return pred, prob


def predict_wav2vec2(audio_path: Path, device: torch.device) -> tuple:
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    ckpt = OUTPUTS_DIR / "transformer_wav2vec2"
    if not ckpt.exists():
        ckpt = "facebook/wav2vec2-base"
    model = AutoModelForAudioClassification.from_pretrained(str(ckpt))
    feature_extractor = AutoFeatureExtractor.from_pretrained(str(ckpt))
    model = model.to(device).eval()

    y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    if len(y) > MAX_AUDIO_SAMPLES:
        start = (len(y) - MAX_AUDIO_SAMPLES) // 2
        y = y[start : start + MAX_AUDIO_SAMPLES]
    elif len(y) < MAX_AUDIO_SAMPLES:
        y = np.pad(y, (0, MAX_AUDIO_SAMPLES - len(y)), mode="constant")
    y = y.astype(np.float32)

    inputs = feature_extractor(
        [y], sampling_rate=SAMPLE_RATE, return_tensors="pt",
        padding=True, truncation=True, max_length=MAX_AUDIO_SAMPLES,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    prob = torch.softmax(out.logits, dim=1)[0, 1].item()
    pred = 1 if prob >= 0.5 else 0
    return pred, prob


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict real vs fake for an audio file.")
    parser.add_argument("audio", type=Path, help="Path to WAV file")
    parser.add_argument("--model", "-m", choices=["wav2vec2", "cnn", "rf"], default="wav2vec2")
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"Error: {args.audio} not found")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_names = ["real", "fake"]

    try:
        if args.model == "rf":
            pred, prob = predict_rf(args.audio)
        elif args.model == "cnn":
            pred, prob = predict_cnn(args.audio, device)
        else:
            pred, prob = predict_wav2vec2(args.audio, device)

        print(f"Prediction: {label_names[pred]} (P(fake)={prob:.3f})")
        return 0
    except FileNotFoundError as e:
        print(f"Error: Model not found. Run the training script for {args.model} first.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
