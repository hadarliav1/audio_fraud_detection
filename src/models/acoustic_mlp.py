"""
MLP on acoustic feature vectors. Baseline = CNN on selected acoustic features only.
Uses BatchNorm + Dropout for regularization.
"""

import torch
import torch.nn as nn


class AcousticMLP(nn.Module):
    """MLP for vector input (selected acoustic features). BatchNorm + Dropout to reduce overfitting."""

    def __init__(
        self,
        n_input: int,
        n_classes: int = 2,
        hidden: tuple = (64, 32),
        dropout: float = 0.5,
    ):
        super().__init__()
        layers = []
        prev = n_input
        for h in hidden:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
