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

# Results and reports (structured pipeline outputs)
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
EDA_DIR = PROJECT_ROOT / "eda"  # EDA figures from notebooks 02.5, 03
# Legacy: scripts expect OUTPUTS_DIR for checkpoints and JSON
OUTPUTS_DIR = RESULTS_DIR

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
RANDOM_SEED = 123
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # Of the remaining (train) set
BATCH_SIZE = 16
MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4  # L2 for CNN/optimizers to reduce overfitting

# Baseline = CNN (MLP) on selected acoustic features only. No mel spectrogram.
# Feature selection (shared by baseline and fusion): top N by univariate AUC, drop if correlation > threshold
TOP_ACOUSTIC_N = 30
CORR_THRESHOLD = 0.85
# Stronger regularization for baseline (small feature set, avoid overfitting)
BASELINE_DROPOUT = 0.6
BASELINE_WEIGHT_DECAY = 1e-3
BASELINE_LEARNING_RATE = 5e-4  # slightly lower than default to avoid overshooting

# Hugging Face models; fine-tuned in train_transformers.py
TRANSFORMER_MODELS = [
    "facebook/hubert-base-ls960",
    "facebook/wav2vec2-base",
    "microsoft/wavlm-base",
    "openai/whisper-tiny",
]

# Noise robustness
SNR_LEVELS_DB = [float("inf"), 20, 10, 5, 0]  # inf = clean (for eval)
SNR_LEVELS_NOISY = [20, 10, 5, 0]  # SNR levels for training on noisy data (excludes clean)
NOISE_TYPES = ["white", "pink", "compression"]
NOISY_DATA_DIR = PROJECT_ROOT / "data" / "noisy"  # Noisy dataset root
