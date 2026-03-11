"""
Configuration for AI-Generated Voice Fraud Detection project.
Paths, constants, and experiment settings.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Data paths
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Audio preprocessing
SAMPLE_RATE = 16000  # Hz; standard for speech, aligns with HF models
TARGET_DURATION_SEC = 5.0  # Clip or pad to this length
TRIM_DB = 25  # dB threshold for silence trimming (librosa.effects.trim)
NORMALIZE_MODE = "peak"  # "peak" or "rms"

# Feature extraction
N_MFCC = 13
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
F0_FMIN = 75
F0_FMAX = 600

# Model training
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # Of the remaining (train) set
BATCH_SIZE = 32
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 1e-4

# Hugging Face models; fine-tuned in train_transformers.py
TRANSFORMER_MODELS = [
    "facebook/hubert-base-ls960",
    "facebook/wav2vec2-base",
    "microsoft/wavlm-base",
    "openai/whisper-tiny",
]

# Noise robustness
SNR_LEVELS_DB = [float("inf"), 20, 10, 5, 0]  # inf = clean
NOISE_TYPES = ["white", "pink", "compression"]
