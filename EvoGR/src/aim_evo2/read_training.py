"""Evo2 embedding extraction and training for read-based samples."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .read_data import ReadSample
from .read_model import ReadAttentionClassifier


class ReadEmbeddingDataset(Dataset):
    def __init__(self, samples: list[ReadSample], embeddings: list[torch.Tensor]):
        if len(samples) != len(embeddings):
            raise ValueError("sample and embedding counts differ")
        self.samples, self.embeddings = samples, embeddings

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"features": self.embeddings[index], "label": self.samples[index].label, "name": self.samples[index].path.name}


def collate_reads(items: list[dict[str, Any]]) -> dict[str, Any]:
    max_reads = max(item["features"].shape[0] for item in items)
    dim = items[0]["features"].shape[1]
    features = torch.zeros(len(items), max_reads, dim, dtype=torch.float32)
    mask = torch.zeros(len(items), max_reads, dtype=torch.bool)
    for index, item in enumerate(items):
        count = item["features"].shape[0]
        features[index, :count] = item["features"]
        mask[index, :count] = True
    return {"features": features, "mask": mask, "label": torch.tensor([item["label"] for item in items], dtype=torch.long), "name": [item["name"] for item in items]}


def _layer_embedding(outputs: Any, layer_name: str) -> torch.Tensor:
    embeddings = outputs[1] if isinstance(outputs, tuple) else outputs.get("embeddings") if isinstance(outputs, dict) else None
    if embeddings is None or layer_name not in embeddings:
        raise KeyError(f"Evo2 did not return layer {layer_name!r}")
    hidden = embeddings[layer_name]
    if hidden.ndim == 2:
        hidden = hidden.unsqueeze(0)
    return hidden


def embed_reads(samples: list[ReadSample], checkpoint: Path, model_name: str, layer_name: str, max_read_length: int) -> tuple[list[torch.Tensor], int]:
    if max_read_length < 1:
        raise ValueError("max_read_length must be positive")
    from evo2 import Evo2

    model = Evo2(model_name, local_path=str(checkpoint))
    model.model.eval()
    result: list[torch.Tensor] = []
    hidden_dim: int | None = None
    with torch.inference_mode():
        for sample in samples:
            sample_reads: list[torch.Tensor] = []
            for read in sample.reads:
                sequence = read[:max_read_length]
                token_ids = model.tokenizer.tokenize(sequence)
                input_ids = torch.tensor(token_ids, dtype=torch.long, device=next(model.model.parameters()).device).unsqueeze(0)
                hidden = _layer_embedding(model(input_ids, return_embeddings=True, layer_names=[layer_name]), layer_name)
                pooled = hidden.mean(dim=1).squeeze(0).float().cpu()
                hidden_dim = int(pooled.shape[-1])
                sample_reads.append(pooled)
            result.append(torch.stack(sample_reads))
    if hidden_dim is None:
        raise ValueError("no reads were embedded")
    return result, hidden_dim


def train_read_classifier(embeddings: list[torch.Tensor], samples: list[ReadSample], num_classes: int, epochs: int, batch_size: int, learning_rate: float, device: str) -> ReadAttentionClassifier:
    model = ReadAttentionClassifier(embeddings[0].shape[-1], num_classes).to(device)
    loader = DataLoader(ReadEmbeddingDataset(samples, embeddings), batch_size=batch_size, shuffle=True, collate_fn=collate_reads)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch["features"].to(device), batch["mask"].to(device))
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            optimizer.step()
    return model
