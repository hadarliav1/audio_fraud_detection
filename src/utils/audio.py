"""
Audio I/O and preprocessing utilities.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
from scipy import signal


def load_audio(path: Path, sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    y, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        n = int(len(y) * sr / file_sr)
        y = signal.resample(y, n)
    return y.astype(np.float32)


def save_audio(path: Path, y: np.ndarray, sr: int = 16000) -> None:
    """Save audio to WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr)


def preprocess_audio(
    y: np.ndarray,
    sr: int,
    target_duration_sec: float,
    trim_db: float = 25,
    normalize_mode: str = "peak",
) -> np.ndarray:
    """
    Preprocess audio: trim silence, normalize, clip or pad to target duration.

    Args:
        y: Audio waveform
        sr: Sample rate
        target_duration_sec: Target length in seconds
        trim_db: dB threshold for silence trimming
        normalize_mode: "peak" or "rms"

    Returns:
        Preprocessed waveform
    """
    # Trim leading/trailing silence (energy-based, avoids librosa numba cache issues)
    rms = np.sqrt(np.mean(y**2))
    threshold = rms * (10 ** (-trim_db / 20))
    if threshold > 0:
        above = np.abs(y) > threshold
        where = np.where(above)[0]
        if len(where) > 0:
            y_trimmed = y[where[0] : where[-1] + 1]
        else:
            y_trimmed = y
    else:
        y_trimmed = y

    # Normalize
    if normalize_mode == "peak":
        if np.max(np.abs(y_trimmed)) > 0:
            y_trimmed = y_trimmed / np.max(np.abs(y_trimmed))
    elif normalize_mode == "rms":
        rms = np.sqrt(np.mean(y_trimmed**2))
        if rms > 0:
            y_trimmed = y_trimmed / rms

    target_len = int(target_duration_sec * sr)
    n = len(y_trimmed)

    if n >= target_len:
        # Center crop
        start = (n - target_len) // 2
        y_out = y_trimmed[start : start + target_len]
    else:
        # Zero-pad to center
        pad_total = target_len - n
        pad_start = pad_total // 2
        pad_end = pad_total - pad_start
        y_out = np.pad(y_trimmed, (pad_start, pad_end), mode="constant", constant_values=0)

    return y_out.astype(np.float32)
