#!/usr/bin/env python3
"""
Train CNN spectrogram model.

Uses speaker-disjoint split (no speaker in both train and test) for realistic evaluation.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BATCH_SIZE,
    HOP_LENGTH,
    LEARNING_RATE,
    MAX_EPOCHS,
    N_FFT,
    N_MELS,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    SAMPLE_RATE,
    TEST_SIZE,
    VAL_SIZE,
)
from src.models.cnn_spectrogram import SpectrogramCNN
from src.utils.dataset import SpectrogramDataset
from src.utils.eval import evaluate_binary
from src.utils.paths import get_audio_paths_with_labels
from src.utils.splits import speaker_disjoint_split


def main() -> int:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = get_audio_paths_with_labels(PROCESSED_DIR)
    if not pairs:
        print("No processed audio. Run: python scripts/run_preprocessing.py")
        return 1

    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    tr_pairs, val_pairs, test_pairs = speaker_disjoint_split(
        pairs, TEST_SIZE, val_ratio, RANDOM_SEED
    )
    if not val_pairs:
        val_pairs = tr_pairs[: len(tr_pairs) // 10]

    train_paths = [p for p, _ in tr_pairs]
    train_labels = [l for _, l in tr_pairs]
    val_paths = [p for p, _ in val_pairs]
    val_labels = [l for _, l in val_pairs]
    test_paths = [p for p, _ in test_pairs]
    test_labels = [l for _, l in test_pairs]

    print(f"Speaker-disjoint: train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")

    max_frames = int(5.0 * SAMPLE_RATE / HOP_LENGTH)

    train_ds = SpectrogramDataset(
        train_paths, train_labels, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, max_frames=max_frames,
    )
    val_ds = SpectrogramDataset(
        val_paths, val_labels, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, max_frames=max_frames,
    )
    test_ds = SpectrogramDataset(
        test_paths, test_labels, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, max_frames=max_frames,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = SpectrogramCNN(n_mels=N_MELS, n_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                pred = model(x).argmax(1).cpu()
                correct += (pred == y).sum().item()
                total += len(y)
        val_acc = correct / total
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), OUTPUTS_DIR / "cnn_spectrogram.pt")
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1} val_acc={val_acc:.3f}")

    model.load_state_dict(torch.load(OUTPUTS_DIR / "cnn_spectrogram.pt", weights_only=True))
    model.eval()
    all_pred, all_prob, all_y = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            prob = torch.softmax(logits, 1)[:, 1].cpu().numpy()
            pred = logits.argmax(1).cpu().numpy()
            all_pred.extend(pred)
            all_prob.extend(prob)
            all_y.extend(y.numpy())

    import numpy as np
    metrics = evaluate_binary(np.array(all_y), np.array(all_pred), np.array(all_prob))
    print("CNN Test:", metrics)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / "cnn_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
