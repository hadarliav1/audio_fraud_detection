#!/usr/bin/env python3
"""
Preprocess raw audio: resample, trim, normalize, clip/pad.
Output saved to data/processed/ preserving directory structure.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    NORMALIZE_MODE,
    PROCESSED_DIR,
    RAW_DATA_DIR,
    SAMPLE_RATE,
    TARGET_DURATION_SEC,
    TRIM_DB,
)
from src.utils.audio import load_audio, preprocess_audio, save_audio
from src.utils.paths import get_audio_paths_with_labels


def main() -> int:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Clear previous run to avoid stale files from old (incorrect) labels
    for subdir in ("real", "fake"):
        d = PROCESSED_DIR / subdir
        if d.exists():
            for f in d.glob("*"):
                f.unlink()

    pairs = get_audio_paths_with_labels(RAW_DATA_DIR)
    if not pairs:
        print("No audio files found. Ensure data is in data/raw/ with real/ and fake/ (or similar) subdirs.")
        return 1

    print(f"Found {len(pairs)} audio files")
    n_real = sum(1 for _, l in pairs if l == 0)
    n_fake = sum(1 for _, l in pairs if l == 1)
    print(f"  Real: {n_real}, Fake: {n_fake}")

    for i, (src_path, label) in enumerate(pairs):
        try:
            y = load_audio(src_path, sr=SAMPLE_RATE)
            y_proc = preprocess_audio(
                y,
                sr=SAMPLE_RATE,
                target_duration_sec=TARGET_DURATION_SEC,
                trim_db=TRIM_DB,
                normalize_mode=NORMALIZE_MODE,
            )
            # Preserve structure: processed/label/filename.wav
            subdir = "real" if label == 0 else "fake"
            dest_path = PROCESSED_DIR / subdir / src_path.name
            dest_path = dest_path.with_suffix(".wav")
            save_audio(dest_path, y_proc, sr=SAMPLE_RATE)
        except Exception as e:
            print(f"Error processing {src_path}: {e}")
            continue

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(pairs)}")

    print(f"Done. Processed files in {PROCESSED_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
