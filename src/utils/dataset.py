"""
PyTorch datasets for audio classification.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


def _spec_augment(mel: np.ndarray, n_mels: int, max_time_mask: int = 15, max_freq_mask: int = 10) -> np.ndarray:
    """Apply simple SpecAugment: zero out one random time band and one random freq band. In-place."""
    F, T = mel.shape
    if max_time_mask > 0 and T > max_time_mask:
        t0 = np.random.randint(0, T - max_time_mask + 1)
        mel[:, t0 : t0 + max_time_mask] = mel.min()
    if max_freq_mask > 0 and F > max_freq_mask:
        f0 = np.random.randint(0, F - max_freq_mask + 1)
        mel[f0 : f0 + max_freq_mask, :] = mel.min()
    return mel


class SpectrogramDataset(Dataset):
    """Dataset of mel-spectrograms from audio paths. Optional SpecAugment for training."""

    def __init__(
        self,
        paths: List[Path],
        labels: List[int],
        sr: int = 16000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        max_frames: Optional[int] = None,
        augment: bool = False,
    ):
        self.paths = paths
        self.labels = labels
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_frames = max_frames
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        y, _ = librosa.load(str(self.paths[idx]), sr=self.sr, mono=True)
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

        if self.max_frames:
            T = mel_db.shape[1]
            if T > self.max_frames:
                start = (T - self.max_frames) // 2
                mel_db = mel_db[:, start : start + self.max_frames]
            elif T < self.max_frames:
                pad = np.zeros((self.n_mels, self.max_frames - T))
                mel_db = np.concatenate([mel_db, pad], axis=1)

        if self.augment:
            mel_db = _spec_augment(mel_db.copy(), self.n_mels)

        x = torch.from_numpy(mel_db).float()
        return x, self.labels[idx]
