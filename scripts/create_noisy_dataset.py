#!/usr/bin/env python3
"""
Stage 8 — Part 1: Create noisy dataset.

Generates noisy versions of audio files for each (noise_type, snr) combination.
Uses the SAME speaker-disjoint split from results/split.json.
Does NOT overwrite the original clean dataset.
Output: data/noisy/{noise_type}_{snr}dB/ with real/ and fake/ subdirs.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    NOISY_DATA_DIR,
    NOISE_TYPES,
    RANDOM_SEED,
    SAMPLE_RATE,
    SNR_LEVELS_NOISY,
)
from src.utils.noise_utils import create_all_noisy_datasets
from src.utils.splits import load_split, SPLIT_FILENAME

RESULTS_DIR = PROJECT_ROOT / "results"


def main() -> int:
    split_path = RESULTS_DIR / SPLIT_FILENAME
    loaded = load_split(split_path)
    if loaded is None:
        print("results/split.json missing. Run: python scripts/train_cnn.py first.")
        return 1

    tr_pairs, val_pairs, test_pairs = loaded
    print(f"Using same split: train={len(tr_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")

    NOISY_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Creating noisy datasets...")
    datasets = create_all_noisy_datasets(
        tr_pairs, val_pairs, test_pairs,
        output_root=NOISY_DATA_DIR,
        noise_types=NOISE_TYPES,
        snr_levels=SNR_LEVELS_NOISY,
        sr=SAMPLE_RATE,
        seed=RANDOM_SEED,
    )

    # Save metadata (orig -> noisy mapping) for each condition
    metadata_dir = NOISY_DATA_DIR / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    for key, data in datasets.items():
        meta_path = metadata_dir / f"{key}.json"
        with open(meta_path, "w") as f:
            json.dump({"root": data["root"], "metadata": data["metadata"]}, f, indent=2)
        print(f"  {key}: {data['root']}")

    print(f"\nSaved metadata to {metadata_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
