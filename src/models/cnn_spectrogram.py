"""
CNN on mel-spectrograms for audio deepfake detection.
"""

import torch
import torch.nn as nn


class SpectrogramCNN(nn.Module):
    """
    CNN for 2D mel-spectrogram input.
    Conv2D -> BatchNorm -> ReLU -> MaxPool blocks, then global pooling and FC.
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        # Input: (B, 1, n_mels, time)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, T, F) -> (B, 1, T, F)
        x = self.conv(x)
        return self.fc(x)
