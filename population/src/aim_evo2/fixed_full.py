from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import fixed_context_length, load_config, resolve_path
from .data import ValidatedData
from .dataset import FixedFullDataset, collate_fixed
from .manifests import assert_manifest_current, embedding_manifest_contract, representation_manifest_contract
from .metrics import compute_metrics
from .model import AttentionSetClassifier
from .splits import make_experiment_folds
from .utils import choose_device, runtime_metadata, set_seed, sha256_file, utc_now_iso, write_json

METRIC_NAMES = ("Accuracy", "MCC", "Macro-AUPRC", "Macro-AUROC", "Macro-F1", "Macro-Precision", "Macro-Recall")


def validate_fixed_full_result_schema(result: dict[str, Any]) -> None:
    """Reject incomplete fixed-full JSON before it can be published."""
    folds = result.get("fold_metrics")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("fixed-full result must contain five fold_metrics entries")
    for index, fold in enumerate(folds):
        metrics = fold.get("metrics") if isinstance(fold, dict) else None
        if not isinstance(metrics, dict) or set(metrics) != set(METRIC_NAMES):
            raise ValueError(f"fold {index}: result must contain all seven overall metrics")
    summary = result.get("fold_summary")
    if not isinstance(summary, dict) or any(not isinstance(summary.get(name), dict) or set(summary[name]) != {"mean", "std"} for name in METRIC_NAMES):
        raise ValueError("fixed-full result must contain mean/std for all seven metrics")
    pooled = result.get("oof_pooled_metrics")
    if not isinstance(pooled, dict) or set(pooled) != set(METRIC_NAMES):
        raise ValueError("fixed-full result must contain seven OOF pooled metrics")
    if not isinstance(result.get("oof_per_class"), list) or len(result["oof_per_class"]) != 5:
        raise ValueError("fixed-full result must contain five OOF per-class rows")


def _predict(model: torch.nn.Module, dataset: FixedFullDataset, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[int] = []
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fixed):
            logits, _ = model(batch["features"].to(device), batch["locus_indices"].to(device), batch["attention_mask"].to(device))
            labels.extend(int(value) for value in batch["label"].tolist())
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.asarray(labels, dtype=np.int64), np.concatenate(probabilities, axis=0)


def _new_model(config: dict[str, Any], input_dim: int, n_aim: int, device: torch.device) -> AttentionSetClassifier:
    model_cfg = config["model"]
    return AttentionSetClassifier(
        input_dim,
        n_aim,
        num_classes=int(model_cfg["num_classes"]),
        projection_dim=int(model_cfg["locus_projection_dim"]),
        attention_hidden_dim=int(model_cfg["attention_hidden_dim"]),
        classifier_hidden_dim=int(model_cfg["classifier_hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)


def _train_one_fold(
    config: dict[str, Any],
    data: ValidatedData,
    representation: torch.Tensor,
    fold: dict[str, list[str]],
    fold_id: int,
    device: torch.device,
    artifact_dir: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    set_seed(int(config["project"]["seed"]) + fold_id)
    train = FixedFullDataset(data.genotypes, data.rsid_order, torch.empty(0), fold["train"], precomputed=representation)
    val = FixedFullDataset(data.genotypes, data.rsid_order, torch.empty(0), fold["val"], precomputed=representation)
    test = FixedFullDataset(data.genotypes, data.rsid_order, torch.empty(0), fold["test"], precomputed=representation)
    model = _new_model(config, int(representation.shape[-1]), data.n_aim, device)
    training = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    loss_fn = torch.nn.CrossEntropyLoss()
    best_score = (-np.inf, np.inf)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        losses: list[float] = []
        for batch in DataLoader(train, batch_size=int(training["batch_size"]), shuffle=True, collate_fn=collate_fixed):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch["features"].to(device), batch["locus_indices"].to(device), batch["attention_mask"].to(device))
            loss = loss_fn(logits, batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_y, val_p = _predict(model, val, int(training["batch_size"]), device)
        val_metrics, _, _ = compute_metrics(val_y, val_p, data.label_mapping)
        score = (float(val_metrics["Macro-F1"]), -float(val_metrics["Macro-AUPRC"]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_metrics": val_metrics})
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(training["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError(f"fold {fold_id}: no best validation state was selected")
    model.load_state_dict(best_state)
    test_y, test_p = _predict(model, test, int(training["batch_size"]), device)
    test_metrics, per_class, _ = compute_metrics(test_y, test_p, data.label_mapping)
    test_metrics = {name: test_metrics[name] for name in METRIC_NAMES}
    fold_payload = {
        "experiment_id": fold_id,
        "train_base_fold_ids": [idx for idx in range(5) if idx not in {fold_id, (fold_id + 1) % 5}],
        "val_base_fold_id": (fold_id + 1) % 5,
        "test_base_fold_id": fold_id,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "best_epoch": best_epoch,
        "metrics": test_metrics,
        "per_class": per_class.to_dict(orient="records"),
    }
    write_json(artifact_dir / f"fold_{fold_id}_history.json", {"experiment_id": fold_id, "history": history})
    write_json(artifact_dir / f"fold_{fold_id}_metrics.json", fold_payload)
    return fold_payload, test_y, test_p


def run_fixed_full_cv(config: dict[str, Any], data: ValidatedData, folds: list[dict[str, list[str]]] | None = None) -> dict[str, Any]:
    """Run fixed-full CV using cached L=101 representations only; never imports Evo2."""
    root = Path(config["_root"])
    context_length = fixed_context_length(config)
    if context_length != 101:
        raise ValueError(f"fixed-full requires L=101, found {context_length}")
    embedding_dir = resolve_path(config, config["artifacts"]["embedding_dir"])
    embedding_path = embedding_dir / "allele_embeddings.pt"
    embedding_manifest_path = embedding_dir / "embedding_manifest.json"
    assert_manifest_current(embedding_manifest_path, embedding_manifest_contract(config, data), "embedding")
    embedding_manifest = json.loads(embedding_manifest_path.read_text(encoding="utf-8"))
    representation_dir = resolve_path(config, config["artifacts"]["representation_dir"])
    assert_manifest_current(representation_dir / "representation_manifest.json", representation_manifest_contract(config, data, embedding_path, int(embedding_manifest["hidden_dim"])), "representation")
    representation_path = representation_dir / f"representations_L{context_length}.npy"
    representation = torch.from_numpy(np.load(representation_path, mmap_mode="r"))
    expected_shape = (len(data.genotypes), data.n_aim, int(embedding_manifest["hidden_dim"]) * 2)
    if tuple(representation.shape) != expected_shape:
        raise ValueError(f"representation shape mismatch: expected {expected_shape}, found {tuple(representation.shape)}")
    if folds is None:
        from .splits import load_outer_folds
        folds = load_outer_folds(resolve_path(config, config["artifacts"]["cv_dir"]) / "folds.json")
    if len(folds) != 5:
        raise ValueError("fixed-full requires exactly five experiment folds")
    device = torch.device(choose_device())
    artifact_dir = resolve_path(config, config["artifacts"]["cv_dir"]) / "fixed_full"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    fold_payloads: list[dict[str, Any]] = []
    oof_y = np.full(len(data.genotypes), -1, dtype=np.int64)
    oof_p = np.full((len(data.genotypes), int(config["model"]["num_classes"])), np.nan, dtype=np.float64)
    positions = {sid: idx for idx, sid in enumerate(data.genotypes["SampleID"])}
    for fold_id, fold in enumerate(folds):
        payload, test_y, test_p = _train_one_fold(config, data, representation, fold, fold_id, device, artifact_dir)
        fold_payloads.append(payload)
        for sid, y, probability in zip(fold["test"], test_y, test_p, strict=True):
            idx = positions[sid]
            if oof_y[idx] != -1:
                raise ValueError(f"OOF duplicate test prediction for SampleID={sid}")
            oof_y[idx], oof_p[idx] = y, probability
    if np.any(oof_y < 0) or not np.isfinite(oof_p).all():
        raise ValueError("OOF predictions do not cover every sample exactly once")
    pooled, pooled_per_class, _ = compute_metrics(oof_y, oof_p, data.label_mapping)
    pooled = {name: pooled[name] for name in METRIC_NAMES}
    fold_summary = {}
    for name in METRIC_NAMES:
        values = np.asarray([fold["metrics"][name] for fold in fold_payloads], dtype=float)
        fold_summary[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    cv_manifest_path = resolve_path(config, config["artifacts"]["cv_dir"]) / "folds.json"
    result = {
        "schema_version": "3.0", "pipeline_name": "fixed_full", "completed": True,
        "x": int(config["data"]["added_aims_per_autosome"]), "n_aim": data.n_aim,
        "flank_bp": int(config["sequence"]["fixed_flank_bp"]), "context_length": context_length,
        "fold_metrics": fold_payloads, "fold_summary": fold_summary,
        "oof_pooled_metrics": pooled, "oof_per_class": pooled_per_class.to_dict(orient="records"),
        "provenance": {
            "created_at_utc": utc_now_iso(), "seed": int(config["project"]["seed"]),
            "runtime": runtime_metadata(int(config["project"]["seed"])),
            "genotype_sha256": sha256_file(resolve_path(config, config["data"]["genotype_csv"])),
            "embedding_manifest_sha256": sha256_file(embedding_manifest_path),
            "representation_manifest_sha256": sha256_file(representation_dir / "representation_manifest.json"),
            "cv_manifest_sha256": sha256_file(cv_manifest_path),
            "model": {"class": "AttentionSetClassifier", **{key: value for key, value in config["model"].items()}},
            "config": {key: value for key, value in config.items() if not key.startswith("_")},
        },
    }
    validate_fixed_full_result_schema(result)
    output_path = resolve_path(config, config["results"]["fixed_full_json"])
    write_json(output_path, result)
    np.save(artifact_dir / "oof_probabilities.npy", oof_p)
    np.save(artifact_dir / "oof_labels.npy", oof_y)
    return result
