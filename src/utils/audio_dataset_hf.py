"""
Dataset for Hugging Face audio models (Wav2Vec2, HuBERT, WavLM, Whisper).
"""

from pathlib import Path
from typing import List

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


class HFAudioDataset(Dataset):
    """Dataset that loads audio and returns waveforms for HF models."""

    def __init__(
        self,
        paths: List[Path],
        labels: List[int],
        sr: int = 16000,
        max_length_sec: float = 5.0,
    ):
        self.paths = paths
        self.labels = labels
        self.sr = sr
        self.max_length = int(max_length_sec * sr)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        y, _ = librosa.load(str(self.paths[idx]), sr=self.sr, mono=True)
        if len(y) > self.max_length:
            start = (len(y) - self.max_length) // 2
            y = y[start : start + self.max_length]
        elif len(y) < self.max_length:
            y = np.pad(y, (0, self.max_length - len(y)), mode="constant")
        return {"array": y.astype(np.float32), "label": self.labels[idx]}
