from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import fixed_context_length, resolve_path, variable_context_lengths
from .data import ValidatedData
from .dataset import (
    FixedFullDataset,
    StaticVariableSNPDataset,
    VariableContextDataset,
    VariableSNPDataset,
    collate_fixed,
    collate_variable_context,
    collate_variable_snp,
    make_static_subset_map,
    resolve_buckets,
)
from .fixed_full import METRIC_NAMES
from .manifests import assert_manifest_current, embedding_manifest_contract, representation_manifest_contract
from .metrics import compute_metrics
from .model import AttentionSetClassifier
from .splits import load_outer_folds
from .utils import choose_device, runtime_metadata, set_seed, sha256_file, utc_now_iso, write_json


def resolve_variable_snp_k_values(config: dict[str, Any], n_aim: int) -> list[int]:
    values = {
        max(int(config["variable_snp"]["min_snps"]), min(n_aim, round(float(fraction) * n_aim)))
        for fraction in config["variable_snp"]["eval_fractions"]
    }
    return sorted(values)


def _model(config: dict[str, Any], input_dim: int, n_aim: int, device: torch.device) -> AttentionSetClassifier:
    model_cfg = config["model"]
    return AttentionSetClassifier(
        input_dim, n_aim, num_classes=int(model_cfg["num_classes"]),
        projection_dim=int(model_cfg["locus_projection_dim"]),
        attention_hidden_dim=int(model_cfg["attention_hidden_dim"]),
        classifier_hidden_dim=int(model_cfg["classifier_hidden_dim"]), dropout=float(model_cfg["dropout"]),
    ).to(device)


def _predict(model: torch.nn.Module, dataset, batch_size: int, device: torch.device, collate) -> tuple[list[str], np.ndarray, np.ndarray]:
    model.eval(); ids: list[str] = []; labels: list[int] = []; probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate):
            logits, _ = model(batch["features"].to(device), batch["locus_indices"].to(device), batch["attention_mask"].to(device))
            ids.extend(str(value) for value in batch["sample_id"]); labels.extend(int(value) for value in batch["label"].tolist())
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return ids, np.asarray(labels), np.concatenate(probabilities)


def _fit(config: dict[str, Any], data: ValidatedData, train_dataset, val_fn, input_dim: int, device: torch.device) -> tuple[AttentionSetClassifier, int, list[dict[str, Any]]]:
    set_seed(int(config["project"]["seed"]))
    model = _model(config, input_dim, data.n_aim, device)
    training = config["training"]; optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    loss_fn = torch.nn.CrossEntropyLoss(); best = (-np.inf, -np.inf); best_state = None; best_epoch = 0; stale = 0; history = []
    collate = collate_variable_snp if isinstance(train_dataset, VariableSNPDataset) else collate_variable_context
    for epoch in range(1, int(training["max_epochs"]) + 1):
        if hasattr(train_dataset, "set_epoch"): train_dataset.set_epoch(epoch)
        model.train(); losses = []
        for batch in DataLoader(train_dataset, batch_size=int(training["batch_size"]), shuffle=True, collate_fn=collate):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch["features"].to(device), batch["locus_indices"].to(device), batch["attention_mask"].to(device))
            loss = loss_fn(logits, batch["label"].to(device)); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"])); optimizer.step(); losses.append(float(loss.detach().cpu()))
        score, val_detail = val_fn(model)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": val_detail})
        if score > best:
            best = score; best_state = copy.deepcopy(model.state_dict()); best_epoch = epoch; stale = 0
        else: stale += 1
        if stale >= int(training["early_stopping_patience"]): break
    if best_state is None: raise RuntimeError("no validation checkpoint selected")
    model.load_state_dict(best_state)
    return model, best_epoch, history


def _aggregate_replicates(ids: list[str], labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(ids): grouped[sample_id].append(index)
    ordered = sorted(grouped)
    y = np.asarray([labels[grouped[sample_id][0]] for sample_id in ordered])
    p = np.asarray([probabilities[grouped[sample_id]].mean(axis=0) for sample_id in ordered])
    return y, p


def _metric_summary(fold_metrics: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {name: {"mean": float(np.mean([m[name] for m in fold_metrics])), "std": float(np.std([m[name] for m in fold_metrics], ddof=0))} for name in METRIC_NAMES}


def _load_representation_inputs(config: dict[str, Any], data: ValidatedData, lengths: list[int]) -> tuple[Path, dict[int, torch.Tensor], dict[str, Any]]:
    root = Path(config["_root"]); embedding_dir = resolve_path(config, config["artifacts"]["embedding_dir"]); embedding_path = embedding_dir / "allele_embeddings.pt"
    embedding_manifest_path = embedding_dir / "embedding_manifest.json"; assert_manifest_current(embedding_manifest_path, embedding_manifest_contract(config, data), "embedding")
    embedding_manifest = json.loads(embedding_manifest_path.read_text(encoding="utf-8")); rep_dir = resolve_path(config, config["artifacts"]["representation_dir"])
    assert_manifest_current(rep_dir / "representation_manifest.json", representation_manifest_contract(config, data, embedding_path, int(embedding_manifest["hidden_dim"])), "representation")
    reps = {length: torch.from_numpy(np.load(rep_dir / f"representations_L{length}.npy", mmap_mode="r")) for length in lengths}
    expected = (len(data.genotypes), data.n_aim, int(embedding_manifest["hidden_dim"]) * 2)
    for length, representation in reps.items():
        if tuple(representation.shape) != expected: raise ValueError(f"representation L={length} shape mismatch: expected {expected}, found {tuple(representation.shape)}")
    return embedding_path, reps, embedding_manifest


def run_variable_snp_cv(config: dict[str, Any], data: ValidatedData, folds: list[dict[str, list[str]]] | None = None) -> dict[str, Any]:
    length = fixed_context_length(config)
    if length != 101: raise ValueError(f"variable-SNP requires context_length=101, found {length}")
    embedding_path, reps, embedding_manifest = _load_representation_inputs(config, data, [length]); representation = reps[length]
    folds = folds or load_outer_folds(resolve_path(config, config["artifacts"]["cv_dir"]) / "folds.json")
    k_values = resolve_variable_snp_k_values(config, data.n_aim); buckets = resolve_buckets(data.n_aim, int(config["variable_snp"]["min_snps"]), config["variable_snp"]["bucket_fractions"])
    device = torch.device(choose_device()); artifact_dir = resolve_path(config, config["artifacts"]["cv_dir"]) / "variable_snp"; artifact_dir.mkdir(parents=True, exist_ok=True)
    fold_outputs = []; oof: dict[int, dict[str, tuple[int, np.ndarray]]] = {k: {} for k in k_values}
    for fold_id, fold in enumerate(folds):
        train = VariableSNPDataset(data.genotypes, data.rsid_order, torch.empty(0), fold["train"], buckets=buckets, seed=int(config["project"]["seed"]), fold_id=fold_id, precomputed=representation)
        val_map = make_static_subset_map(fold["val"], data.n_aim, k_values, int(config["variable_snp"]["val_replicates_per_k"]), int(config["project"]["seed"]), fold_id, "val")
        test_map = make_static_subset_map(fold["test"], data.n_aim, k_values, int(config["variable_snp"]["test_replicates_per_k"]), int(config["project"]["seed"]), fold_id, "test")
        val_datasets = {k: StaticVariableSNPDataset(data.genotypes, data.rsid_order, torch.empty(0), fold["val"], subsets=val_map, k_values=[k], replicates=int(config["variable_snp"]["val_replicates_per_k"]), precomputed=representation) for k in k_values}
        def validate(model):
            details = {}; scores = []
            for k, dataset in val_datasets.items():
                ids, y_rep, p_rep = _predict(model, dataset, int(config["training"]["batch_size"]), device, collate_variable_snp); y, p = _aggregate_replicates(ids, y_rep, p_rep); metrics, _, _ = compute_metrics(y, p, data.label_mapping); details[str(k)] = metrics; scores.append(metrics["Macro-F1"])
            return (float(np.mean(scores)), float(np.mean([details[str(k)]["Macro-AUPRC"] for k in k_values])), details)
        def val_fn(model):
            score, tie, details = validate(model); return (score, tie), details
        model, best_epoch, history = _fit(config, data, train, val_fn, int(representation.shape[-1]), device)
        by_k = {}; test_datasets = {k: StaticVariableSNPDataset(data.genotypes, data.rsid_order, torch.empty(0), fold["test"], subsets=test_map, k_values=[k], replicates=int(config["variable_snp"]["test_replicates_per_k"]), precomputed=representation) for k in k_values}
        for k, dataset in test_datasets.items():
            ids, y_rep, p_rep = _predict(model, dataset, int(config["training"]["batch_size"]), device, collate_variable_snp); y, p = _aggregate_replicates(ids, y_rep, p_rep); metrics, per_class, _ = compute_metrics(y, p, data.label_mapping); metrics = {name: metrics[name] for name in METRIC_NAMES}; by_k[str(k)] = {"metrics": metrics, "per_class": per_class.to_dict(orient="records")}
            for sample_id, label, probability in zip(sorted(set(ids)), y, p, strict=True): oof[k][sample_id] = (int(label), probability)
        payload = {"experiment_id": fold_id, "best_epoch": best_epoch, "train_base_fold_ids": [i for i in range(5) if i not in {fold_id, (fold_id + 1) % 5}], "val_base_fold_id": (fold_id + 1) % 5, "test_base_fold_id": fold_id, "metrics_by_k": by_k}
        fold_outputs.append(payload); write_json(artifact_dir / f"fold_{fold_id}_history.json", {"history": history}); write_json(artifact_dir / f"fold_{fold_id}_metrics.json", payload)
    metrics_by_k = {}; fold_metrics_by_k = {}
    for k in k_values:
        if len(oof[k]) != len(data.genotypes):
            raise ValueError(f"OOF coverage mismatch for K={k}: expected {len(data.genotypes)}, found {len(oof[k])}")
        pooled_y = np.asarray([value[0] for value in oof[k].values()]); pooled_p = np.asarray([value[1] for value in oof[k].values()]); pooled, per_class, _ = compute_metrics(pooled_y, pooled_p, data.label_mapping); metrics_by_k[str(k)] = {"metrics": {name: pooled[name] for name in METRIC_NAMES}, "per_class": per_class.to_dict(orient="records")}; fold_metrics_by_k[str(k)] = _metric_summary([fold["metrics_by_k"][str(k)]["metrics"] for fold in fold_outputs])
    result = {"schema_version": "3.0", "pipeline_name": "variable_snp", "completed": True, "x": int(config["data"]["added_aims_per_autosome"]), "n_aim": data.n_aim, "flank_bp": 50, "context_length": 101, "k_values": k_values, "buckets": [list(bucket) for bucket in buckets], "metrics_by_k": metrics_by_k, "fold_metrics_by_k": fold_metrics_by_k, "folds": fold_outputs, "overall_oof_summary": metrics_by_k, "provenance": _provenance(config, data, embedding_path, embedding_manifest, "variable_snp", {"k_values": k_values, "buckets": [list(bucket) for bucket in buckets]})}
    write_json(resolve_path(config, config["results"]["variable_snp_json"]), result); return result


def run_variable_context_cv(config: dict[str, Any], data: ValidatedData, folds: list[dict[str, list[str]]] | None = None) -> dict[str, Any]:
    train_lengths = variable_context_lengths(config, "train"); eval_lengths = variable_context_lengths(config, "eval")
    lengths = sorted(set(train_lengths) | set(eval_lengths)); embedding_path, reps, embedding_manifest = _load_representation_inputs(config, data, lengths); folds = folds or load_outer_folds(resolve_path(config, config["artifacts"]["cv_dir"]) / "folds.json")
    device = torch.device(choose_device()); artifact_dir = resolve_path(config, config["artifacts"]["cv_dir"]) / "variable_context"; artifact_dir.mkdir(parents=True, exist_ok=True); fold_outputs = []; oof = {length: {} for length in eval_lengths}
    deltas = {length: torch.empty(0) for length in lengths}
    for fold_id, fold in enumerate(folds):
        train = VariableContextDataset(data.genotypes, data.rsid_order, deltas, train_lengths, seed=int(config["project"]["seed"]), fold_id=fold_id, sample_ids=fold["train"], precomputed_by_length=reps)
        val_datasets = {length: FixedFullDataset(data.genotypes, data.rsid_order, torch.empty(0), fold["val"], precomputed=reps[length]) for length in eval_lengths}
        def val_fn(model):
            details = {}; scores = []
            for length, dataset in val_datasets.items():
                ids, y, p = _predict(model, dataset, int(config["training"]["batch_size"]), device, collate_fixed); metrics, _, _ = compute_metrics(y, p, data.label_mapping); details[str(length)] = metrics; scores.append(metrics["Macro-F1"])
            return (float(np.mean(scores)), float(np.mean([details[str(length)]["Macro-AUPRC"] for length in eval_lengths]))), details
        model, best_epoch, history = _fit(config, data, train, val_fn, int(reps[lengths[0]].shape[-1]), device); by_length = {}; test_datasets = {length: FixedFullDataset(data.genotypes, data.rsid_order, torch.empty(0), fold["test"], precomputed=reps[length]) for length in eval_lengths}
        for length, dataset in test_datasets.items():
            ids, y, p = _predict(model, dataset, int(config["training"]["batch_size"]), device, collate_fixed); metrics, per_class, _ = compute_metrics(y, p, data.label_mapping); by_length[str(length)] = {"metrics": {name: metrics[name] for name in METRIC_NAMES}, "per_class": per_class.to_dict(orient="records")}; oof[length].update({sid: (int(label), probability) for sid, label, probability in zip(ids, y, p, strict=True)})
        payload = {"experiment_id": fold_id, "best_epoch": best_epoch, "train_base_fold_ids": [i for i in range(5) if i not in {fold_id, (fold_id + 1) % 5}], "val_base_fold_id": (fold_id + 1) % 5, "test_base_fold_id": fold_id, "metrics_by_context_length": by_length}; fold_outputs.append(payload); write_json(artifact_dir / f"fold_{fold_id}_history.json", {"history": history}); write_json(artifact_dir / f"fold_{fold_id}_metrics.json", payload)
    metrics_by_context = {}; fold_metrics_by_context = {}
    for length in eval_lengths:
        if len(oof[length]) != len(data.genotypes):
            raise ValueError(f"OOF coverage mismatch for L={length}: expected {len(data.genotypes)}, found {len(oof[length])}")
        y = np.asarray([value[0] for value in oof[length].values()]); p = np.asarray([value[1] for value in oof[length].values()]); pooled, per_class, _ = compute_metrics(y, p, data.label_mapping); metrics_by_context[str(length)] = {"metrics": {name: pooled[name] for name in METRIC_NAMES}, "per_class": per_class.to_dict(orient="records")}; fold_metrics_by_context[str(length)] = _metric_summary([fold["metrics_by_context_length"][str(length)]["metrics"] for fold in fold_outputs])
    result = {"schema_version": "3.0", "pipeline_name": "variable_context", "completed": True, "x": int(config["data"]["added_aims_per_autosome"]), "n_aim": data.n_aim, "train_n": [int(n) for n in config["variable_context"]["train_flank_bp"]], "eval_n": [int(n) for n in config["variable_context"]["eval_flank_bp"]], "train_flank_bp": [int(n) for n in config["variable_context"]["train_flank_bp"]], "eval_flank_bp": [int(n) for n in config["variable_context"]["eval_flank_bp"]], "train_context_lengths": train_lengths, "eval_context_lengths": eval_lengths, "metrics_by_context_length": metrics_by_context, "fold_metrics_by_context_length": fold_metrics_by_context, "folds": fold_outputs, "overall_oof_summary": metrics_by_context, "provenance": _provenance(config, data, embedding_path, embedding_manifest, "variable_context", {"train_n": config["variable_context"]["train_flank_bp"], "eval_n": config["variable_context"]["eval_flank_bp"], "train_context_lengths": train_lengths, "eval_context_lengths": eval_lengths})}
    write_json(resolve_path(config, config["results"]["variable_context_json"]), result); return result


def _provenance(config, data, embedding_path, embedding_manifest, pipeline, details):
    rep_manifest = resolve_path(config, config["artifacts"]["representation_dir"]) / "representation_manifest.json"; cv_manifest = resolve_path(config, config["artifacts"]["cv_dir"]) / "folds.json"
    return {"created_at_utc": utc_now_iso(), "pipeline": pipeline, "seed": int(config["project"]["seed"]), "x": int(config["data"]["added_aims_per_autosome"]), "n_aim": data.n_aim, "genotype_sha256": sha256_file(resolve_path(config, config["data"]["genotype_csv"])), "embedding_manifest_sha256": sha256_file(embedding_path.parent / "embedding_manifest.json"), "representation_manifest_sha256": sha256_file(rep_manifest), "cv_manifest_sha256": sha256_file(cv_manifest), "model": config["model"], "details": details, "runtime": runtime_metadata(int(config["project"]["seed"]))}
