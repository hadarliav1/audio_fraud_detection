"""
Train/val/test splitting utilities. Speaker-disjoint split for fair evaluation.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

SPLIT_FILENAME = "split.json"


def get_speaker(path: Path) -> str:
    """Extract speaker/source ID from path for disjoint split.

    We make this conservative and explicit so that we don't
    accidentally treat 'real'/'fake' or other tokens as speakers.

    Current convention (matches preprocessing scripts):
      - Files are stored under .../processed/{real,fake}/<stem>.wav
      - The stem encodes speaker as the first token before '_' or '-'.
        Examples:
          data/processed/real/4214_10.wav           -> speaker '4214'
          data/processed/fake/00004528-00000002.wav -> speaker '00004528'
    """
    stem = Path(path).stem
    spk = stem.split("_")[0].split("-")[0]
    # Guard against accidentally returning label/empty tokens
    if spk.lower() in {"real", "fake", ""}:
        # Fall back to parent folder name, which in our data is the
        # label folder ('real'/'fake'); prepend stem to keep it unique.
        spk = f"{stem}:{Path(path).parent.name}"
    return spk


def speaker_disjoint_split(
    pairs: List[Tuple[Path, int]],
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Split (path, label) pairs by speaker so no speaker appears in both train and test.
    Pairs are sorted by path first so the split is deterministic across scripts (CNN,
    transformers, fusion, noise) for the same data and seed.

    Args:
        pairs: List of (path, label)
        test_ratio: Fraction of speakers for test
        val_ratio: Fraction of train speakers for val
        seed: Random seed

    Returns:
        tr_pairs, val_pairs, test_pairs
    """
    pairs = sorted(pairs, key=lambda x: str(x[0]))
    rng = np.random.default_rng(seed)
    speakers: dict = {}
    for path, label in pairs:
        spk = get_speaker(path)
        if spk not in speakers:
            speakers[spk] = []
        speakers[spk].append((path, label))

    spk_list = np.array(sorted(speakers.keys()))
    rng.shuffle(spk_list)
    n_test = max(1, int(len(spk_list) * test_ratio))
    test_speakers = set(spk_list[:n_test])
    train_speakers = set(spk_list[n_test:])

    train_pairs = [p for spk in train_speakers for p in speakers[spk]]
    test_pairs = [p for spk in test_speakers for p in speakers[spk]]

    # Stratify val by class so val has both real and fake (needed for AUC and threshold tuning)
    def speaker_label(spk: str) -> int:
        return speakers[spk][0][1]  # label of first sample (speakers usually single-class)

    train_spk_list = np.array(list(train_speakers))
    spk_by_label: dict = {0: [], 1: []}
    for spk in train_spk_list:
        lab = speaker_label(spk)
        if lab in spk_by_label:
            spk_by_label[lab].append(spk)

    n_val = max(0, int(len(train_spk_list) * val_ratio))
    val_speakers = set()
    if n_val > 0 and spk_by_label[0] and spk_by_label[1]:
        rng.shuffle(spk_by_label[0])
        rng.shuffle(spk_by_label[1])
        n_val_0 = max(1, min(len(spk_by_label[0]), n_val // 2))
        n_val_1 = min(len(spk_by_label[1]), n_val - n_val_0)
        if n_val_1 <= 0:
            n_val_0 = min(len(spk_by_label[0]), n_val)
            n_val_1 = 0
        val_speakers = set(spk_by_label[0][:n_val_0]) | set(spk_by_label[1][:n_val_1])
    elif n_val > 0:
        rng.shuffle(train_spk_list)
        val_speakers = set(train_spk_list[:n_val])

    tr_speakers = set(train_spk_list) - val_speakers
    tr_pairs = [p for spk in tr_speakers for p in speakers[spk]]
    val_pairs = [p for spk in val_speakers for p in speakers[spk]] if val_speakers else []

    # Sanity checks: speaker-disjoint property
    def _spks(pairs):
        return {get_speaker(p) for p, _ in pairs}

    tr_spk = _spks(tr_pairs)
    val_spk = _spks(val_pairs)
    test_spk = _spks(test_pairs)
    assert tr_spk.isdisjoint(val_spk), "Train/val speakers overlap"
    assert tr_spk.isdisjoint(test_spk), "Train/test speakers overlap"
    assert val_spk.isdisjoint(test_spk), "Val/test speakers overlap"

    return tr_pairs, val_pairs, test_pairs


def save_split(
    tr_pairs: List[Tuple[Path, int]],
    val_pairs: List[Tuple[Path, int]],
    test_pairs: List[Tuple[Path, int]],
    path: Path,
) -> None:
    """Save train/val/test split to JSON so other scripts (transformers, fusion, noise) use the same split."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "train_paths": [str(p) for p, _ in tr_pairs],
        "train_labels": [int(l) for _, l in tr_pairs],
        "val_paths": [str(p) for p, _ in val_pairs],
        "val_labels": [int(l) for _, l in val_pairs],
        "test_paths": [str(p) for p, _ in test_pairs],
        "test_labels": [int(l) for _, l in test_pairs],
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def load_split(path: Path) -> Optional[Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]]:
    """Load train/val/test split from JSON. Returns (tr_pairs, val_pairs, test_pairs) or None if missing."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    tr_pairs = [(Path(p), int(l)) for p, l in zip(data["train_paths"], data["train_labels"])]
    val_pairs = [(Path(p), int(l)) for p, l in zip(data["val_paths"], data["val_labels"])]
    test_pairs = [(Path(p), int(l)) for p, l in zip(data["test_paths"], data["test_labels"])]
    return tr_pairs, val_pairs, test_pairs


def summarize_split(
    tr_pairs: List[Tuple[Path, int]],
    val_pairs: List[Tuple[Path, int]],
    test_pairs: List[Tuple[Path, int]],
    path: Path,
) -> None:
    """Write basic diagnostics about the split to JSON.

    Includes per-split:
      - #speakers
      - #samples
      - label distribution
      - speaker overlap flags
    """

    def _stats(pairs: List[Tuple[Path, int]]) -> dict:
        speakers = {get_speaker(p) for p, _ in pairs}
        labels = [l for _, l in pairs]
        return {
            "n_speakers": len(speakers),
            "n_samples": len(pairs),
            "label_counts": {
                "real": int(sum(l == 0 for l in labels)),
                "fake": int(sum(l == 1 for l in labels)),
            },
        }

    def _spks(pairs: List[Tuple[Path, int]]) -> set:
        return {get_speaker(p) for p, _ in pairs}

    report = {
        "train": _stats(tr_pairs),
        "val": _stats(val_pairs),
        "test": _stats(test_pairs),
        "overlap": {
            "train_val": bool(_spks(tr_pairs) & _spks(val_pairs)),
            "train_test": bool(_spks(tr_pairs) & _spks(test_pairs)),
            "val_test": bool(_spks(val_pairs) & _spks(test_pairs)),
        },
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
