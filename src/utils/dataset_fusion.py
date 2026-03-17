from pathlib import Path
from typing import List, Dict, Any

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from config import SAMPLE_RATE, TARGET_DURATION_SEC


class FusionDataset(Dataset):
    """
    Dataset that returns:
      input_values, attention_mask, acoustic, labels
    """

    def __init__(
        self,
        paths: List[Path],
        labels: List[int],
        df_acoustic,
        scaler: StandardScaler,
        feature_cols: List[str],
        feature_extractor,
    ):
        assert len(paths) == len(labels)
        self.paths = [Path(p) for p in paths]
        self.labels = labels
        self.df = df_acoustic
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.fe = feature_extractor
        self.max_samples = int(TARGET_DURATION_SEC * SAMPLE_RATE)

        self.path_to_idx: Dict[str, int] = {}
        for idx, row in self.df.iterrows():
            p = Path(row["path"]).resolve()
            self.path_to_idx[str(p)] = idx
            self.path_to_idx[p.as_posix()] = idx

    def __len__(self) -> int:
        return len(self.paths)

    def _load_audio(self, path: Path) -> np.ndarray:
        y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        if len(y) > self.max_samples:
            start = (len(y) - self.max_samples) // 2
            y = y[start : start + self.max_samples]
        elif len(y) < self.max_samples:
            y = np.pad(y, (0, self.max_samples - len(y)), mode="constant")
        return y.astype(np.float32)

    def _get_acoustic(self, path: Path) -> np.ndarray:
        key = str(path.resolve())
        if key not in self.path_to_idx:
            key = path.resolve().as_posix()
        idx = self.path_to_idx[key]
        row = self.df.iloc[idx][self.feature_cols].values.astype(np.float64).reshape(1, -1)
        row = self.scaler.transform(row)
        return row.astype(np.float32)[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.paths[idx]
        label = self.labels[idx]

        y = self._load_audio(path)
        inputs = self.fe(
            [y],
            sampling_rate=SAMPLE_RATE,
            max_length=self.max_samples,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_values = inputs["input_values"][0]          # (T,)

        # Some feature extractors don't return attention_mask; fall back to all ones.
        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"][0]
        else:
            attention_mask = torch.ones_like(input_values, dtype=torch.long)

        ac = self._get_acoustic(path)

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "acoustic": torch.from_numpy(ac),
            "labels": torch.tensor(label, dtype=torch.long),
        }