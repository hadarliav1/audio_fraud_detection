#!/usr/bin/env python3
"""
Extract acoustic features from processed audio and save to data/features/.

Usage:
  python scripts/extract_features.py                    # Extract from data/processed
  python scripts/extract_features.py --input-dir X --output Y   # Custom dir -> csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    FEATURES_DIR,
    F0_FMAX,
    F0_FMIN,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    N_MFCC,
    PROCESSED_DIR,
    SAMPLE_RATE,
)
from src.features.acoustic_scipy import extract_all_features, features_to_vector, get_feature_names
from src.utils.audio import load_audio
from src.utils.paths import get_audio_paths_with_labels


def extract_features_from_dir(input_dir: Path, output_csv: Path) -> int:
    """Extract acoustic features from all audio in input_dir, save to output_csv."""
    pairs = get_audio_paths_with_labels(input_dir)
    if not pairs:
        print(f"No audio found in {input_dir}")
        return 1

    print(f"Extracting features from {len(pairs)} files in {input_dir}...")
    rows = []
    for i, (path, label) in enumerate(pairs):
        try:
            y = load_audio(path, sr=SAMPLE_RATE)
            feat = extract_all_features(
                y, sr=SAMPLE_RATE,
                n_mfcc=N_MFCC, n_mels=N_MELS, hop_length=HOP_LENGTH,
                n_fft=N_FFT, f0_fmin=F0_FMIN, f0_fmax=F0_FMAX,
            )
            vec = features_to_vector(feat)
            row = {"path": str(path), "label": label}
            for j, name in enumerate(get_feature_names()):
                row[name] = vec[j]
            rows.append(row)
        except Exception as e:
            print(f"  Error {path}: {e}")
            continue
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(pairs)}")

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, help="Custom input directory")
    parser.add_argument("--output", type=Path, help="Output CSV path")
    args = parser.parse_args()

    if args.input_dir is not None and args.output is not None:
        return extract_features_from_dir(args.input_dir, args.output)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    pairs = get_audio_paths_with_labels(PROCESSED_DIR)
    if not pairs:
        print("No processed audio found. Run: python scripts/run_preprocessing.py")
        return 1

    print(f"Extracting features from {len(pairs)} files...")

    rows = []
    for i, (path, label) in enumerate(pairs):
        try:
            y = load_audio(path, sr=SAMPLE_RATE)
            feat = extract_all_features(
                y,
                sr=SAMPLE_RATE,
                n_mfcc=N_MFCC,
                n_mels=N_MELS,
                hop_length=HOP_LENGTH,
                n_fft=N_FFT,
                f0_fmin=F0_FMIN,
                f0_fmax=F0_FMAX,
            )
            vec = features_to_vector(feat)
            row = {"path": str(path), "label": label}
            for j, name in enumerate(get_feature_names()):
                row[name] = vec[j]
            rows.append(row)
        except Exception as e:
            print(f"Error {path}: {e}")
            continue

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(pairs)}")

    df = pd.DataFrame(rows)
    out_path = FEATURES_DIR / "acoustic_features.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
