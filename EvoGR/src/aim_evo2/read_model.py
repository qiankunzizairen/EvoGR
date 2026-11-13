"""Read-level attention classifier used by the text-sample pipeline."""
from __future__ import annotations

import torch
from torch import nn


class ReadAttentionClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, projection_dim: int = 128, attention_hidden_dim: int = 64, classifier_hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.read_projection = nn.Sequential(nn.Linear(input_dim, projection_dim), nn.GELU(), nn.LayerNorm(projection_dim))
        self.read_attention = nn.Sequential(nn.Linear(projection_dim, attention_hidden_dim), nn.Tanh(), nn.Linear(attention_hidden_dim, 1))
        self.classifier = nn.Sequential(nn.LayerNorm(projection_dim), nn.Linear(projection_dim, classifier_hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(classifier_hidden_dim, num_classes))

    def forward(self, read_features: torch.Tensor, read_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if read_features.ndim != 3 or read_mask.shape != read_features.shape[:2]:
            raise ValueError("read_features must be [B,R,D] and read_mask must be [B,R]")
        if not read_mask.any(dim=1).all():
            raise ValueError("each sample must contain at least one read")
        projected = self.read_projection(read_features)
        scores = self.read_attention(projected).squeeze(-1)
        weights = torch.softmax(scores.masked_fill(~read_mask, float("-inf")), dim=1)
        weights = torch.where(read_mask, weights, torch.zeros_like(weights))
        pooled = torch.sum(weights.unsqueeze(-1) * projected, dim=1)
        return self.classifier(pooled), weights
