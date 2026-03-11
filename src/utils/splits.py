"""
Train/val/test splitting utilities. Speaker-disjoint split for fair evaluation.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np


def get_speaker(path: Path) -> str:
    """Extract speaker/source ID from path for disjoint split.
    Convention: stem split by '_' and '-', first part = speaker.
    """
    stem = Path(path).stem
    return stem.split("_")[0].split("-")[0]


def speaker_disjoint_split(
    pairs: List[Tuple[Path, int]],
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Split (path, label) pairs by speaker so no speaker appears in both train and test.

    Args:
        pairs: List of (path, label)
        test_ratio: Fraction of speakers for test
        val_ratio: Fraction of train speakers for val
        seed: Random seed

    Returns:
        tr_pairs, val_pairs, test_pairs
    """
    rng = np.random.default_rng(seed)
    speakers: dict = {}
    for path, label in pairs:
        spk = get_speaker(path)
        if spk not in speakers:
            speakers[spk] = []
        speakers[spk].append((path, label))

    spk_list = np.array(list(speakers.keys()))
    rng.shuffle(spk_list)
    n_test = max(1, int(len(spk_list) * test_ratio))
    test_speakers = set(spk_list[:n_test])
    train_speakers = set(spk_list[n_test:])

    train_pairs = [p for spk in train_speakers for p in speakers[spk]]
    test_pairs = [p for spk in test_speakers for p in speakers[spk]]

    train_spk_list = np.array(list(train_speakers))
    rng.shuffle(train_spk_list)
    n_val = max(0, int(len(train_spk_list) * val_ratio))
    val_speakers = set(train_spk_list[:n_val])
    tr_speakers = set(train_spk_list[n_val:])

    tr_pairs = [p for spk in tr_speakers for p in speakers[spk]]
    val_pairs = [p for spk in val_speakers for p in speakers[spk]] if n_val > 0 else []

    return tr_pairs, val_pairs, test_pairs
