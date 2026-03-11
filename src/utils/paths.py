"""
Path and label utilities for audio datasets.
"""

from pathlib import Path
from typing import List, Tuple

# Labels: 0 = real/bonafide, 1 = fake/spoof
# NOTE: Do NOT use "0" or "1" - they false-match filenames like 10.wav, 101.wav
REAL_LABELS = {"real", "bonafide", "genuine", "human"}
FAKE_LABELS = {"fake", "spoof", "synthetic", "ai", "deepfake"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Top-level folders that are known to be real (adarshsingh0903 Kaggle dataset)
REAL_FOLDERS = {"real_samples", "real", "bonafide"}
# All other top-level folders in this dataset are TTS models = fake
TTS_FAKE_FOLDERS = frozenset({
    "naturalspeech3", "flashspeech", "seedtts_files", "valle",
    "prompttts2", "xtts", "voicebox", "openai",
})


def get_audio_paths_with_labels(
    root: Path,
    real_labels: set = None,
    fake_labels: set = None,
) -> List[Tuple[Path, int]]:
    """
    Scan directory for audio files and infer labels from folder names.

    For adarshsingh0903/audio-deepfake-detection-dataset:
    - real_samples/ = real (bonafide)
    - NaturalSpeech3, FlashSpeech, OpenAI, etc. = fake (TTS)

    Returns:
        List of (path, label) where label is 0 (real) or 1 (fake)
    """
    real_labels = real_labels or REAL_LABELS
    fake_labels = fake_labels or FAKE_LABELS
    results: List[Tuple[Path, int]] = []

    for path in root.rglob("*"):
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        rel_path = path.relative_to(root)
        parts = [p.lower() for p in rel_path.parts]
        top_folder = parts[0] if parts else ""

        label = None

        # 1. Dataset-specific: use top-level folder (avoids "0"/"1" in filenames)
        if top_folder in REAL_FOLDERS:
            label = 0
        elif top_folder in TTS_FAKE_FOLDERS:
            label = 1

        # 2. Fallback: match folder names in path (not full path string)
        if label is None:
            path_str = rel_path.as_posix().lower()
            if any(r in path_str for r in real_labels):
                label = 0
            elif any(f in path_str for f in fake_labels):
                label = 1

        if label is not None:
            results.append((path, label))

    return results
