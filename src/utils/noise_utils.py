"""
Noise robustness utilities for Stage 8.

- corrupt_audio: add white, pink, or compression noise at target SNR
- create_noisy_dataset: generate noisy versions of audio files preserving split
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal

from src.utils.audio import load_audio, save_audio


def add_white_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise to achieve target SNR (dB)."""
    if snr_db == float("inf"):
        return y.copy()
    sig_power = np.mean(y**2)
    if sig_power < 1e-12:
        return y.copy()
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
    sig_power = np.mean(y**2)
    if sig_power < 1e-12:
        return y.copy()
    noise_power = sig_power / (10 ** (snr_db / 10))
    return y + pink * np.sqrt(noise_power)


def add_compression_noise(
    y: np.ndarray, snr_db: float, sr: int, rng: np.random.Generator
) -> np.ndarray:
    """Simulate codec artifacts: bandwidth limit + quantization. SNR controls noise level."""
    if snr_db == float("inf"):
        return y.copy()
    # Downsample to 8kHz and back (telephony-like) using scipy to avoid librosa/numba issues
    n_8k = int(len(y) * 8000 / sr)
    n_8k = max(1, n_8k)
    y_8k = signal.resample(y.astype(np.float64), n_8k)
    n_back = int(len(y_8k) * sr / 8000)
    y_back = signal.resample(y_8k, n_back)
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


def corrupt_audio(
    y: np.ndarray,
    noise_type: str,
    snr_db: float,
    sr: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply noise to audio. Returns copy if snr_db is inf (clean)."""
    if noise_type == "white":
        return add_white_noise(y, snr_db, rng)
    if noise_type == "pink":
        return add_pink_noise(y, snr_db, rng)
    if noise_type == "compression":
        return add_compression_noise(y, snr_db, sr, rng)
    return y.copy()


def get_noisy_subdir_name(noise_type: str, snr_db: float) -> str:
    """e.g. white_20dB, pink_10dB"""
    return f"{noise_type}_{int(snr_db)}dB"


def create_noisy_dataset(
    tr_pairs: List[Tuple[Path, int]],
    val_pairs: List[Tuple[Path, int]],
    test_pairs: List[Tuple[Path, int]],
    noise_type: str,
    snr_db: float,
    output_root: Path,
    sr: int,
    seed: int,
) -> Tuple[Path, Dict[str, List[Tuple[str, str]]]]:
    """
    Create noisy dataset for one (noise_type, snr_db) condition.

    Applies noise to each file and saves to output_root/{noise_type}_{snr}dB/real|fake/.
    Preserves speaker identity and labels. Does NOT overwrite original data.

    Returns:
        - Path to the noisy dataset root (output_root/{noise_type}_{snr}dB)
        - metadata: {"train": [(orig, noisy), ...], "val": [...], "test": [...]}
    """
    rng = np.random.default_rng(seed)
    subdir_name = get_noisy_subdir_name(noise_type, snr_db)
    noisy_root = output_root / subdir_name
    metadata: Dict[str, List[Tuple[str, str]]] = {"train": [], "val": [], "test": []}

    for split_name, pairs in [("train", tr_pairs), ("val", val_pairs), ("test", test_pairs)]:
        for orig_path, label in pairs:
            orig_path = Path(orig_path)
            try:
                y = load_audio(orig_path, sr=sr)
            except Exception as e:
                raise RuntimeError(f"Failed to load {orig_path}: {e}") from e

            y_noisy = corrupt_audio(y.astype(np.float32), noise_type, snr_db, sr, rng)

            # Use label for folder: real (0) or fake (1) - ensures get_audio_paths_with_labels works
            label_folder = "real" if label == 0 else "fake"

            out_dir = noisy_root / label_folder
            out_path = out_dir / orig_path.name

            save_audio(out_path, y_noisy, sr)
            metadata[split_name].append((str(orig_path.resolve()), str(out_path.resolve())))

    return noisy_root, metadata


def create_all_noisy_datasets(
    tr_pairs: List[Tuple[Path, int]],
    val_pairs: List[Tuple[Path, int]],
    test_pairs: List[Tuple[Path, int]],
    output_root: Path,
    noise_types: List[str],
    snr_levels: List[float],
    sr: int,
    seed: int,
) -> Dict[str, Dict]:
    """
    Create noisy datasets for all (noise_type, snr) combinations.

    Returns dict: {(noise_type, snr): {"root": Path, "metadata": {...}}}
    """
    results = {}
    for noise_type in noise_types:
        for snr_db in snr_levels:
            key = f"{noise_type}_{int(snr_db)}dB"
            root, metadata = create_noisy_dataset(
                tr_pairs, val_pairs, test_pairs,
                noise_type, snr_db, output_root, sr, seed
            )
            results[key] = {"root": str(root), "metadata": metadata}
    return results
