"""
Acoustic feature extraction using scipy/numpy only (no librosa).
Avoids librosa/numba cache errors on some systems.
"""

from typing import Dict

import numpy as np
from scipy import signal
from scipy.fft import dct


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700 * (10 ** (mel / 2595) - 1)


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595 * np.log10(1 + hz / 700)


def _mel_filterbank(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    """Create mel filterbank matrix."""
    low_freq = 0
    high_freq = sr / 2
    low_mel = _hz_to_mel(np.array([low_freq]))[0]
    high_mel = _hz_to_mel(np.array([high_freq]))[0]
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        for j in range(left, center):
            fbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            fbank[i, j] = (right - j) / (right - center)
    return fbank


def _safe_mean(x: np.ndarray) -> float:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.mean(x)) if len(x) else 0.0


def _safe_std(x: np.ndarray) -> float:
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
    """Extract acoustic features using scipy/numpy only."""
    # STFT
    f, t, Z = signal.stft(y, sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    S = np.abs(Z) ** 2
    S = np.nan_to_num(S, nan=1e-12, posinf=1e10, neginf=0)
    S = np.clip(S, 1e-12, 1e10)

    # Mel spectrogram
    fbank = _mel_filterbank(n_mels, n_fft, sr)
    mel_spec = fbank @ S
    mel_spec = np.clip(mel_spec, 1e-10, None)

    # MFCCs (DCT of log mel)
    log_mel = np.log(mel_spec + 1e-10)
    mfcc = dct(log_mel, axis=0, type=2, norm="ortho")[:n_mfcc]
    mfcc_mean = [_safe_mean(mfcc[i]) for i in range(n_mfcc)]
    mfcc_std = [_safe_std(mfcc[i]) for i in range(n_mfcc)]

    # Delta MFCCs
    mfcc_delta = np.gradient(mfcc, axis=1)
    mfcc_delta_mean = [_safe_mean(mfcc_delta[i]) for i in range(n_mfcc)]
    mfcc_delta_std = [_safe_std(mfcc_delta[i]) for i in range(n_mfcc)]

    # Delta-delta
    mfcc_delta2 = np.gradient(mfcc_delta, axis=1)
    mfcc_delta2_mean = [_safe_mean(mfcc_delta2[i]) for i in range(n_mfcc)]
    mfcc_delta2_std = [_safe_std(mfcc_delta2[i]) for i in range(n_mfcc)]

    # Spectral flux
    spectral_flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    spectral_flux_mean = _safe_mean(spectral_flux)
    spectral_flux_std = _safe_std(spectral_flux)

    # F0 jitter (autocorrelation-based pitch, simplified)
    try:
        corr = np.correlate(y, y, mode="full")
        corr = corr[len(corr) // 2 :]
        min_period = int(sr / f0_fmax)
        max_period = int(sr / f0_fmin)
        search = corr[min_period:max_period]
        if len(search) > 1:
            peaks = signal.find_peaks(search)[0]
            if len(peaks) > 1:
                periods = np.diff(peaks) + min_period
                f0_est = sr / np.mean(periods)
                jitter = np.std(periods) / (np.mean(periods) + 1e-8)
                f0_jitter = float(jitter)
            else:
                f0_jitter = 0.0
        else:
            f0_jitter = 0.0
    except Exception:
        f0_jitter = 0.0

    # Spectral rolloff
    freqs = np.arange(S.shape[0], dtype=float) * sr / n_fft
    cumsum = np.cumsum(S, axis=0)
    thresh = 0.85 * (cumsum[-1, :] + 1e-10)
    rolloff = np.array([
        freqs[min(np.searchsorted(cumsum[:, i], thresh[i]), len(freqs) - 1)]
        for i in range(S.shape[1])
    ])
    spectral_rolloff_mean = _safe_mean(rolloff)
    spectral_rolloff_std = _safe_std(rolloff)

    # Spectral centroid
    centroid = np.sum(freqs[:, None] * S, axis=0) / (np.sum(S, axis=0) + 1e-10)
    spectral_centroid_mean = _safe_mean(centroid)
    spectral_centroid_std = _safe_std(centroid)

    # HNR (simplified: harmonic/percussive ratio via autocorrelation)
    try:
        rms_all = np.sqrt(np.mean(y**2))
        rms_harm = np.sqrt(np.mean(y**2)) * (np.max(corr[:max_period]) / (corr[0] + 1e-10))
        hnr_mean = float(rms_harm / (rms_all - rms_harm + 1e-10))
        hnr_std = 0.0
    except Exception:
        hnr_mean, hnr_std = 0.0, 0.0

    # RMS
    frame_len = hop_length * 2
    rms = np.array([np.sqrt(np.mean(y[i * hop_length : i * hop_length + frame_len] ** 2))
                    for i in range((len(y) - frame_len) // hop_length + 1)])
    rms = rms[: S.shape[1]]
    rms_mean = _safe_mean(rms)
    rms_std = _safe_std(rms)

    # Spectral bandwidth
    bandwidth = np.sqrt(np.sum(((freqs[:, None] - centroid) ** 2) * S, axis=0) / (np.sum(S, axis=0) + 1e-10))
    spectral_bandwidth_mean = _safe_mean(bandwidth)
    spectral_bandwidth_std = _safe_std(bandwidth)

    # ZCR (simple frame-based)
    n_frames = S.shape[1]
    zcr = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        end = min(start + frame_len, len(y))
        if end > start + 1:
            frame = y[start:end]
            zcr[i] = np.mean(np.abs(np.diff(np.sign(frame)))) / 2
    zcr_mean = _safe_mean(zcr)
    zcr_std = _safe_std(zcr)

    temporal_spectral_flux = _safe_std(spectral_flux)

    # Chroma (simplified: 12 bands of spectrum)
    n_chroma = 12
    n_freq = S.shape[0]
    chroma = np.zeros((n_chroma, S.shape[1]))
    for c in range(n_chroma):
        start = (c * n_freq) // n_chroma
        end = ((c + 1) * n_freq) // n_chroma
        if end > start:
            chroma[c, :] = np.sum(S[start:end, :], axis=0)
    chroma = chroma / (np.sum(chroma, axis=0, keepdims=True) + 1e-10)
    chroma_mean = [_safe_mean(chroma[i]) for i in range(12)]
    chroma_std = [_safe_std(chroma[i]) for i in range(12)]

    out = {
        "mfcc_mean": mfcc_mean, "mfcc_std": mfcc_std,
        "mfcc_delta_mean": mfcc_delta_mean, "mfcc_delta_std": mfcc_delta_std,
        "mfcc_delta2_mean": mfcc_delta2_mean, "mfcc_delta2_std": mfcc_delta2_std,
        "spectral_flux_mean": spectral_flux_mean, "spectral_flux_std": spectral_flux_std,
        "f0_jitter": f0_jitter,
        "spectral_rolloff_mean": spectral_rolloff_mean, "spectral_rolloff_std": spectral_rolloff_std,
        "spectral_centroid_mean": spectral_centroid_mean, "spectral_centroid_std": spectral_centroid_std,
        "hnr_mean": hnr_mean, "hnr_std": hnr_std,
        "rms_mean": rms_mean, "rms_std": rms_std,
        "spectral_bandwidth_mean": spectral_bandwidth_mean, "spectral_bandwidth_std": spectral_bandwidth_std,
        "zcr_mean": zcr_mean, "zcr_std": zcr_std,
        "temporal_spectral_flux": temporal_spectral_flux,
        "chroma_mean": chroma_mean, "chroma_std": chroma_std,
    }
    if include_raw:
        out["mfcc_2d"] = mfcc.T
        out["chroma_2d"] = chroma.T
    return out


def features_to_vector(feat: Dict) -> np.ndarray:
    """Convert feature dict to flat vector."""
    parts = []
    for k in [
        "mfcc_mean", "mfcc_std", "mfcc_delta_mean", "mfcc_delta_std",
        "mfcc_delta2_mean", "mfcc_delta2_std",
        "spectral_flux_mean", "spectral_flux_std", "f0_jitter",
        "spectral_rolloff_mean", "spectral_rolloff_std",
        "spectral_centroid_mean", "spectral_centroid_std",
        "hnr_mean", "hnr_std", "rms_mean", "rms_std",
        "spectral_bandwidth_mean", "spectral_bandwidth_std",
        "zcr_mean", "zcr_std", "temporal_spectral_flux",
        "chroma_mean", "chroma_std",
    ]:
        v = feat[k]
        parts.extend(v) if isinstance(v, (list, np.ndarray)) else parts.append(v)
    return np.array(parts, dtype=np.float32)


def get_feature_names() -> list:
    """Return ordered feature names."""
    names = [f"mfcc_mean_{i+1}" for i in range(13)] + [f"mfcc_std_{i+1}" for i in range(13)]
    names += [f"mfcc_delta_mean_{i+1}" for i in range(13)] + [f"mfcc_delta_std_{i+1}" for i in range(13)]
    names += [f"mfcc_delta2_mean_{i+1}" for i in range(13)] + [f"mfcc_delta2_std_{i+1}" for i in range(13)]
    names += ["spectral_flux_mean", "spectral_flux_std", "f0_jitter",
              "spectral_rolloff_mean", "spectral_rolloff_std",
              "spectral_centroid_mean", "spectral_centroid_std",
              "hnr_mean", "hnr_std", "rms_mean", "rms_std",
              "spectral_bandwidth_mean", "spectral_bandwidth_std",
              "zcr_mean", "zcr_std", "temporal_spectral_flux"]
    names += [f"chroma_mean_{i+1}" for i in range(12)] + [f"chroma_std_{i+1}" for i in range(12)]
    return names
