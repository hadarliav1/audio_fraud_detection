"""
Acoustic feature extraction for audio deepfake detection.

Features:
1. MFCCs (1–13) mean and std
2. Delta MFCCs (1–13) mean and std — temporal dynamics
3. Delta-delta MFCCs (1–13) mean and std — acceleration
4. Spectral Flux
5. F0 Jitter
6. Spectral Rolloff
7. Spectral Centroid — brightness
8. HNR (Harmonic-to-Noise Ratio)
9. RMS Energy
10. Spectral Bandwidth
11. Zero-Crossing Rate — high-freq content
12. Temporal Spectral Flux variation
13. Chromagram
"""

from typing import Dict, Optional

import librosa
import numpy as np


def _safe_mean(x: np.ndarray) -> float:
    """Mean, ignoring NaN/inf."""
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.mean(x)) if len(x) else 0.0


def _safe_std(x: np.ndarray) -> float:
    """Std, ignoring NaN/inf."""
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.std(x)) if len(x) > 1 else 0.0


def extract_all_features(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_mels: int = 128,
    hop_length: int = 512,
    n_fft: int = 2048,
    f0_fmin: float = 75,
    f0_fmax: float = 600,
    include_raw: bool = False,
) -> Dict:
    """
    Extract all acoustic features from waveform.

    Args:
        y: Audio waveform
        sr: Sample rate
        n_mfcc, n_mels, hop_length, n_fft: librosa params
        f0_fmin, f0_fmax: F0 range for jitter/HNR
        include_raw: If True, include mfcc_2d and chroma_2d for heatmaps

    Returns:
        Dict with scalar/vector features; if include_raw, also mfcc_2d, chroma_2d
    """
    # Spectrogram
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    mel_spec = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=n_mels)

    # 1. MFCCs (1–13) mean and std
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=n_mfcc, sr=sr)
    mfcc_mean = [_safe_mean(mfcc[i]) for i in range(n_mfcc)]
    mfcc_std = [_safe_std(mfcc[i]) for i in range(n_mfcc)]

    # 2. Delta MFCCs (temporal dynamics)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_mean = [_safe_mean(mfcc_delta[i]) for i in range(n_mfcc)]
    mfcc_delta_std = [_safe_std(mfcc_delta[i]) for i in range(n_mfcc)]

    # 3. Delta-delta MFCCs (acceleration)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_delta2_mean = [_safe_mean(mfcc_delta2[i]) for i in range(n_mfcc)]
    mfcc_delta2_std = [_safe_std(mfcc_delta2[i]) for i in range(n_mfcc)]

    # 4. Spectral Flux (frame-to-frame spectral change)
    spectral_flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    spectral_flux_mean = _safe_mean(spectral_flux)
    spectral_flux_std = _safe_std(spectral_flux)

    # 3. F0 Jitter (cycle-to-cycle F0 variation)
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=f0_fmin, fmax=f0_fmax, sr=sr, hop_length=hop_length
    )
    f0_voiced = f0[voiced_flag]
    if len(f0_voiced) > 1:
        jitter = np.abs(np.diff(f0_voiced))
        f0_jitter = _safe_mean(jitter) / (np.mean(f0_voiced) + 1e-8)  # relative jitter
    else:
        f0_jitter = 0.0

    # 4. Spectral Rolloff (frequency below which 85% of energy lies)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
    spectral_rolloff_mean = _safe_mean(rolloff[0])
    spectral_rolloff_std = _safe_std(rolloff[0])

    # 5. HNR (Harmonic-to-Noise Ratio) — approximate via autocorrelation
    try:
        hnr = librosa.effects.harmonic(y, margin=8)
        nhr = librosa.effects.percussive(y, margin=8)
        hnr_ratio = np.sqrt(np.mean(hnr**2)) / (np.sqrt(np.mean(nhr**2)) + 1e-8)
        hnr_mean = float(hnr_ratio)
        hnr_std = 0.0  # single value per file
    except Exception:
        hnr_mean, hnr_std = 0.0, 0.0

    # 6. RMS Energy
    rms = librosa.feature.rms(S=S)[0]
    rms_mean = _safe_mean(rms)
    rms_std = _safe_std(rms)

    # 7. Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    spectral_bandwidth_mean = _safe_mean(bandwidth)
    spectral_bandwidth_std = _safe_std(bandwidth)

    # 8. Spectral Centroid (brightness)
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    spectral_centroid_mean = _safe_mean(centroid)
    spectral_centroid_std = _safe_std(centroid)

    # 9. Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length, frame_length=n_fft)[0]
    zcr_mean = _safe_mean(zcr)
    zcr_std = _safe_std(zcr)

    # 10. Temporal Spectral Flux variation (std of spectral flux over time)
    temporal_spectral_flux = _safe_std(spectral_flux)

    # 11. MFCC heatmap — per-frame MFCCs (returned for visualization)
    mfcc_2d = mfcc.T  # (n_frames, n_mfcc)

    # 12. Chromagram (pitch-class distribution)
    chroma = librosa.feature.chroma_stft(S=S, sr=sr, n_fft=n_fft)
    chroma_mean = [_safe_mean(chroma[i]) for i in range(12)]
    chroma_std = [_safe_std(chroma[i]) for i in range(12)]

    out = {
        "mfcc_mean": mfcc_mean,
        "mfcc_std": mfcc_std,
        "mfcc_delta_mean": mfcc_delta_mean,
        "mfcc_delta_std": mfcc_delta_std,
        "mfcc_delta2_mean": mfcc_delta2_mean,
        "mfcc_delta2_std": mfcc_delta2_std,
        "spectral_flux_mean": spectral_flux_mean,
        "spectral_flux_std": spectral_flux_std,
        "f0_jitter": f0_jitter,
        "spectral_rolloff_mean": spectral_rolloff_mean,
        "spectral_rolloff_std": spectral_rolloff_std,
        "spectral_centroid_mean": spectral_centroid_mean,
        "spectral_centroid_std": spectral_centroid_std,
        "hnr_mean": hnr_mean,
        "hnr_std": hnr_std,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "spectral_bandwidth_mean": spectral_bandwidth_mean,
        "spectral_bandwidth_std": spectral_bandwidth_std,
        "zcr_mean": zcr_mean,
        "zcr_std": zcr_std,
        "temporal_spectral_flux": temporal_spectral_flux,
        "chroma_mean": chroma_mean,
        "chroma_std": chroma_std,
    }
    if include_raw:
        out["mfcc_2d"] = mfcc_2d
        out["chroma_2d"] = chroma.T  # (n_frames, 12)
    return out


def features_to_vector(feat: Dict) -> np.ndarray:
    """Convert feature dict to flat numpy vector for ML."""
    parts = []
    for k in [
        "mfcc_mean", "mfcc_std",
        "mfcc_delta_mean", "mfcc_delta_std",
        "mfcc_delta2_mean", "mfcc_delta2_std",
        "spectral_flux_mean", "spectral_flux_std",
        "f0_jitter",
        "spectral_rolloff_mean", "spectral_rolloff_std",
        "spectral_centroid_mean", "spectral_centroid_std",
        "hnr_mean", "hnr_std",
        "rms_mean", "rms_std",
        "spectral_bandwidth_mean", "spectral_bandwidth_std",
        "zcr_mean", "zcr_std",
        "temporal_spectral_flux",
        "chroma_mean", "chroma_std",
    ]:
        v = feat[k]
        if isinstance(v, (list, np.ndarray)):
            parts.extend(v)
        else:
            parts.append(v)
    return np.array(parts, dtype=np.float32)


def get_feature_names() -> list:
    """Return ordered feature names for the vector."""
    names = []
    for i in range(13):
        names.append(f"mfcc_mean_{i+1}")
    for i in range(13):
        names.append(f"mfcc_std_{i+1}")
    for i in range(13):
        names.append(f"mfcc_delta_mean_{i+1}")
    for i in range(13):
        names.append(f"mfcc_delta_std_{i+1}")
    for i in range(13):
        names.append(f"mfcc_delta2_mean_{i+1}")
    for i in range(13):
        names.append(f"mfcc_delta2_std_{i+1}")
    names.extend([
        "spectral_flux_mean", "spectral_flux_std",
        "f0_jitter",
        "spectral_rolloff_mean", "spectral_rolloff_std",
        "spectral_centroid_mean", "spectral_centroid_std",
        "hnr_mean", "hnr_std",
        "rms_mean", "rms_std",
        "spectral_bandwidth_mean", "spectral_bandwidth_std",
        "zcr_mean", "zcr_std",
        "temporal_spectral_flux",
    ])
    for i in range(12):
        names.append(f"chroma_mean_{i+1}")
    for i in range(12):
        names.append(f"chroma_std_{i+1}")
    return names
