from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from .data import ValidatedData
from .dataset import AimRepresentationDataset
from .model import AttentionSetClassifier
from .splits import make_inner_validation_split
from .utils import set_seed, sha256_file, write_json


def _make_model(input_dim: int, num_loci: int, config: dict) -> AttentionSetClassifier:
    return AttentionSetClassifier(
        input_dim=input_dim,
        num_loci=num_loci,
        num_classes=int(config["num_classes"]),
        projection_dim=int(config["locus_projection_dim"]),
        attention_hidden_dim=int(config["attention_hidden_dim"]),
        classifier_hidden_dim=int(config["classifier_hidden_dim"]),
        dropout=float(config["dropout"]),
    )


def _run_epoch(model, loader, optimizer, device, clip_norm: float | None = None) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    training = optimizer is not None
    model.train(training)
    losses = []
    predictions = []
    labels = []
    for batch in loader:
        features = batch["features"].to(device)
        y = batch["label"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits, _ = model(features, batch.get("locus_indices", None).to(device) if batch.get("locus_indices") is not None else None, batch.get("attention_mask", None).to(device) if batch.get("attention_mask") is not None else None)
        loss = criterion(logits, y)
        if training:
            loss.backward()
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        predictions.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
        labels.extend(y.detach().cpu().numpy().tolist())
    return float(sum(losses) / max(len(losses), 1)), float(f1_score(labels, predictions, average="macro", zero_division=0))


def train_holdout_head(
    data: ValidatedData,
    deltas: torch.Tensor,
    train_samples: pd.DataFrame,
    config: dict,
    output_dir: Path,
    history_path: Path,
    device: str,
    split_manifest_path: Path,
    embedding_manifest_path: Path | None = None,
) -> int:
    seed = int(config["project"]["seed"])
    set_seed(seed)
    inner_train_ids, inner_val_ids = make_inner_validation_split(
        train_samples,
        float(config["split"]["inner_validation_fraction_of_outer_train"]),
        seed,
    )
    input_dim = int(deltas.shape[1] * 2)
    model = _make_model(input_dim, len(data.rsid_order), config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    batch_size = int(config["training"]["batch_size"])
    from .dataset import collate_fixed
    train_loader = DataLoader(AimRepresentationDataset(data.genotypes, data.rsid_order, deltas, inner_train_ids), batch_size=batch_size, shuffle=True, collate_fn=collate_fixed)
    val_loader = DataLoader(AimRepresentationDataset(data.genotypes, data.rsid_order, deltas, inner_val_ids), batch_size=batch_size, shuffle=False, collate_fn=collate_fixed)
    best_epoch = 1
    best_f1 = -1.0
    best_loss = float("inf")
    patience = int(config["training"]["early_stopping_patience"])
    stale = 0
    history = []
    for epoch in range(1, int(config["training"]["max_epochs"]) + 1):
        train_loss, train_f1 = _run_epoch(model, train_loader, optimizer, device, float(config["training"]["gradient_clip_norm"]))
        val_loss, val_f1 = _run_epoch(model, val_loader, None, device)
        history.append({"phase": "inner", "epoch": epoch, "train_loss": train_loss, "train_macro_f1": train_f1, "val_loss": val_loss, "val_macro_f1": val_f1})
        improved = val_f1 > best_f1 or (val_f1 == best_f1 and val_loss < best_loss)
        if improved:
            best_epoch, best_f1, best_loss, stale = epoch, val_f1, val_loss, 0
        else:
            stale += 1
        if stale >= patience:
            break
    set_seed(seed)
    final_model = _make_model(input_dim, len(data.rsid_order), config["model"]).to(device)
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    full_loader = DataLoader(AimRepresentationDataset(data.genotypes, data.rsid_order, deltas, train_samples["SampleID"].tolist()), batch_size=batch_size, shuffle=True, collate_fn=collate_fixed)
    for epoch in range(1, best_epoch + 1):
        train_loss, train_f1 = _run_epoch(final_model, full_loader, final_optimizer, device, float(config["training"]["gradient_clip_norm"]))
        history.append({"phase": "outer_train", "epoch": epoch, "train_loss": train_loss, "train_macro_f1": train_f1, "val_loss": "", "val_macro_f1": ""})
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "population_head.pt"
    torch.save({"model_state_dict": final_model.state_dict(), "best_epoch": best_epoch}, model_path)
    pd.DataFrame(history).to_csv(history_path, index=False)
    manifest = {
        "seed": seed,
        "rsid_order": data.rsid_order,
        "label_mapping": {str(k): v for k, v in data.label_mapping.items()},
        "model_architecture": "AttentionSetClassifier",
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "embedding_manifest_sha256": sha256_file(embedding_manifest_path) if embedding_manifest_path and embedding_manifest_path.exists() else None,
        "best_epoch": best_epoch,
        "optimizer": "AdamW",
        "learning_rate": float(config["training"]["learning_rate"]),
        "weight_decay": float(config["training"]["weight_decay"]),
    }
    write_json(output_dir / "population_head_manifest.json", manifest)
    return best_epoch
