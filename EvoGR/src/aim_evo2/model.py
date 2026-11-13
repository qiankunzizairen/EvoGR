from __future__ import annotations

import torch
from torch import nn


class AttentionSetClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_loci: int,
        num_classes: int = 5,
        projection_dim: int = 128,
        attention_hidden_dim: int = 64,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.num_loci = num_loci
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.GELU(),
            nn.LayerNorm(projection_dim),
        )
        self.locus_identity = nn.Embedding(num_loci, projection_dim)
        self.attention = nn.Sequential(
            nn.Linear(projection_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(projection_dim),
            nn.Linear(projection_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, locus_features: torch.Tensor, locus_ids: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if locus_features.ndim != 3:
            raise ValueError("locus_features must have shape [B, L, D]")
        batch_size, num_loci, _ = locus_features.shape
        if locus_ids is None:
            locus_ids = torch.arange(num_loci, device=locus_features.device)
        if locus_ids.ndim == 1:
            locus_ids = locus_ids.unsqueeze(0).expand(batch_size, -1)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, num_loci), dtype=torch.bool, device=locus_features.device)
        if attention_mask.shape != (batch_size, num_loci) or not attention_mask.any(dim=1).all():
            raise ValueError("attention_mask must match [B,K] and retain at least one locus per sample")
        if torch.any(locus_ids < 0) or torch.any(locus_ids >= self.num_loci):
            raise ValueError("locus_ids contain an index outside the configured AIM panel")
        projected = self.projection(locus_features)
        x = projected + self.locus_identity(locus_ids)
        scores = self.attention(x).squeeze(-1)
        attention_weights = torch.softmax(scores.masked_fill(~attention_mask, float("-inf")), dim=1)
        attention_weights = torch.where(attention_mask, attention_weights, torch.zeros_like(attention_weights))
        embedding = torch.sum(attention_weights.unsqueeze(-1) * x, dim=1)
        logits = self.classifier(embedding)
        return logits, attention_weights
