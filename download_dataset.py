#!/usr/bin/env python3
"""
Download the Audio Deepfake Detection dataset using kagglehub.
Places files in data/raw/ for use by the preprocessing pipeline.
"""

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_DIR


def main() -> int:
    try:
        import kagglehub
    except ImportError:
        print("Error: kagglehub not installed. Run: pip install kagglehub")
        return 1

    print("Downloading dataset via kagglehub...")
    print("(First run may prompt for Kaggle login)")
    path = kagglehub.dataset_download("adarshsingh0903/audio-deepfake-detection-dataset")
    path = Path(path)
    print(f"Downloaded to cache: {path}")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        dest = RAW_DATA_DIR / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    print(f"Copied to: {RAW_DATA_DIR}")
    print("Done. Run: python scripts/run_preprocessing.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
