from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from torch.utils.data import DataLoader

from .data import ValidatedData
from .dataset import AimRepresentationDataset
from .metrics import compute_metrics
from .model import AttentionSetClassifier
from .utils import write_json


def predict_model(model: AttentionSetClassifier, loader: DataLoader, device: str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    sample_ids: list[str] = []
    labels: list[int] = []
    probabilities = []
    attentions = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            logits, attention = model(features, batch.get("locus_indices", None).to(device) if batch.get("locus_indices") is not None else None, batch.get("attention_mask", None).to(device) if batch.get("attention_mask") is not None else None)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            sample_ids.extend(batch["sample_id"])
            labels.extend(batch["label"].numpy().tolist())
            probabilities.append(probs)
            attentions.append(attention.cpu().numpy())
    return sample_ids, np.asarray(labels), np.concatenate(probabilities), np.concatenate(attentions)


def write_evaluation_outputs(
    output_dir: Path,
    sample_ids: list[str],
    y_true: np.ndarray,
    probabilities: np.ndarray,
    attention: np.ndarray,
    data: ValidatedData,
    rsid_order: list[str],
    prefix: str = "",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall, per_class, matrix = compute_metrics(y_true, probabilities, data.label_mapping)
    write_json(output_dir / f"{prefix}test_metrics.json", overall)
    per_class.to_csv(output_dir / f"{prefix}per_class_metrics.csv", index=False)
    y_pred = probabilities.argmax(axis=1)
    prediction_rows = {
        "SampleID": sample_ids,
        "TrueLabel": y_true,
        "TruePopulation": [data.label_mapping[int(label)] for label in y_true],
        "PredLabel": y_pred,
        "PredPopulation": [data.label_mapping[int(label)] for label in y_pred],
    }
    for label in sorted(data.label_mapping):
        prediction_rows[f"Prob_Label{label}"] = probabilities[:, label]
    pd.DataFrame(prediction_rows).to_csv(output_dir / f"{prefix}test_predictions.csv", index=False)
    _write_attention_outputs(output_dir, sample_ids, y_true, attention, data.label_mapping, rsid_order, prefix)
    _plot_confusion(matrix, sorted(data.label_mapping), output_dir / f"{prefix}confusion_matrix.png")
    _plot_curves(y_true, probabilities, sorted(data.label_mapping), output_dir, prefix)
    return overall


def _write_attention_outputs(output_dir: Path, sample_ids: list[str], labels: np.ndarray, attention: np.ndarray, label_mapping: dict[int, str], rsid_order: list[str], prefix: str) -> None:
    rows = []
    for sample_index, sample_id in enumerate(sample_ids):
        for locus_index, rsid in enumerate(rsid_order):
            rows.append(
                {
                    "SampleID": sample_id,
                    "Label": int(labels[sample_index]),
                    "PopulationName": label_mapping[int(labels[sample_index])],
                    "rsID": rsid,
                    "AttentionWeight": float(attention[sample_index, locus_index]),
                }
            )
    by_sample = pd.DataFrame(rows)
    by_sample.to_csv(output_dir / f"{prefix}attention_weights_by_sample.tsv", sep="\t", index=False)
    by_class = (
        by_sample.groupby(["Label", "PopulationName", "rsID"])["AttentionWeight"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "MeanAttentionWeight", "std": "SDAttentionWeight"})
    )
    by_class["RankWithinClass"] = by_class.groupby("Label")["MeanAttentionWeight"].rank(method="first", ascending=False).astype(int)
    by_class.to_csv(output_dir / f"{prefix}attention_weights_by_class.tsv", sep="\t", index=False)
    global_df = (
        by_sample.groupby("rsID")["AttentionWeight"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "MeanAttentionWeight", "std": "SDAttentionWeight"})
    )
    global_df["GlobalRank"] = global_df["MeanAttentionWeight"].rank(method="first", ascending=False).astype(int)
    global_df.to_csv(output_dir / f"{prefix}attention_weights_global.tsv", sep="\t", index=False)


def _plot_confusion(matrix: np.ndarray, labels: list[int], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_curves(y_true: np.ndarray, probabilities: np.ndarray, labels: list[int], output_dir: Path, prefix: str) -> None:
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4), dpi=160)
    fig_pr, ax_pr = plt.subplots(figsize=(5, 4), dpi=160)
    for label in labels:
        binary = (y_true == label).astype(int)
        if len(set(binary)) < 2:
            continue
        RocCurveDisplay.from_predictions(binary, probabilities[:, label], ax=ax_roc, name=f"Label {label}")
        PrecisionRecallDisplay.from_predictions(binary, probabilities[:, label], ax=ax_pr, name=f"Label {label}")
    ax_roc.set_title("One-vs-rest ROC")
    ax_pr.set_title("One-vs-rest PR")
    fig_roc.tight_layout()
    fig_pr.tight_layout()
    fig_roc.savefig(output_dir / f"{prefix}roc_curves.png")
    fig_pr.savefig(output_dir / f"{prefix}pr_curves.png")
    plt.close(fig_roc)
    plt.close(fig_pr)


def evaluate_checkpoint(
    model_path: Path,
    output_dir: Path,
    data: ValidatedData,
    deltas: torch.Tensor,
    test_sample_ids: list[str],
    model_config: dict,
    device: str,
) -> dict:
    input_dim = int(deltas.shape[1] * 2)
    model = AttentionSetClassifier(
        input_dim=input_dim,
        num_loci=len(data.rsid_order),
        num_classes=int(model_config["num_classes"]),
        projection_dim=int(model_config["locus_projection_dim"]),
        attention_hidden_dim=int(model_config["attention_hidden_dim"]),
        classifier_hidden_dim=int(model_config["classifier_hidden_dim"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    from .dataset import collate_fixed
    dataset = AimRepresentationDataset(data.genotypes, data.rsid_order, deltas, test_sample_ids)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate_fixed)
    sample_ids, y_true, probabilities, attention = predict_model(model, loader, device)
    return write_evaluation_outputs(output_dir, sample_ids, y_true, probabilities, attention, data, data.rsid_order)
