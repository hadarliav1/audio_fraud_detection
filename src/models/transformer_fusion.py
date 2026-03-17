import torch
import torch.nn as nn

from transformers import AutoModel


class TransformerFusionModel(nn.Module):
    """
    End-to-end fusion: transformer encoder + acoustic features.

    Forward expects:
      input_values: (B, T)
      attention_mask: (B, T)
      acoustic: (B, acoustic_dim)
    """

    def __init__(self, base_model_name: str, acoustic_dim: int, dropout: float = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden = self.encoder.config.hidden_size

        self.acoustic_proj = nn.Linear(acoustic_dim, acoustic_dim)

        fusion_in = hidden + acoustic_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, input_values, attention_mask, acoustic):
        out = self.encoder(input_values=input_values, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state  # (B, T, H)

        # Masked mean-pool only when mask length matches encoder sequence length.
        if (
            attention_mask is not None
            and attention_mask.dim() == 2
            and attention_mask.shape[1] == last_hidden.shape[1]
        ):
            mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
            last_hidden = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        else:
            # Either no mask or mismatched length; fallback to simple mean over time.
            last_hidden = last_hidden.mean(1)

        ac = self.acoustic_proj(acoustic)
        fused = torch.cat([last_hidden, ac], dim=1)
        logits = self.classifier(fused)
        return logits