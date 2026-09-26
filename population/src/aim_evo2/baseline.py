from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import ValidatedData, genotype_to_dosage
from .fixed_full import METRIC_NAMES
from .metrics import compute_metrics
from .splits import load_outer_folds
from .config import resolve_path
from .utils import runtime_metadata, sha256_file, utc_now_iso, write_json


def dosage_matrix(frame: pd.DataFrame, rsid_order: list[str]) -> np.ndarray:
    rows = []
    for _, row in frame.iterrows():
        rows.append([genotype_to_dosage(row[rsid]) for rsid in rsid_order])
    return np.asarray(rows, dtype=float)


def run_logistic_regression_baseline(data: ValidatedData, train_ids: list[str], test_ids: list[str], output_dir: Path, seed: int) -> dict:
    indexed = data.genotypes.set_index("SampleID")
    train = indexed.loc[train_ids].reset_index()
    test = indexed.loc[test_ids].reset_index()
    x_train = dosage_matrix(train, data.rsid_order)
    y_train = train["Label"].to_numpy(dtype=int)
    x_test = dosage_matrix(test, data.rsid_order)
    y_test = test["Label"].to_numpy(dtype=int)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)
    classes = list(pipeline.named_steps["logreg"].classes_)
    ordered = np.zeros((len(test), len(data.label_mapping)), dtype=float)
    for source_index, label in enumerate(classes):
        ordered[:, int(label)] = probabilities[:, source_index]
    attention_placeholder = np.full((len(test), len(data.rsid_order)), 1.0 / len(data.rsid_order), dtype=float)
    return write_evaluation_outputs(
        output_dir,
        test["SampleID"].tolist(),
        y_test,
        ordered,
        attention_placeholder,
        data,
        data.rsid_order,
        prefix="",
    )


def run_logreg_5cv(config: dict, data: ValidatedData, folds: list[dict[str, list[str]]] | None = None) -> dict:
    """Run dosage-only multinomial logistic regression on the shared 5-fold mapping."""
    if folds is None:
        folds = load_outer_folds(resolve_path(config, config["artifacts"]["cv_dir"]) / "folds.json")
    if len(folds) != 5:
        raise ValueError("baseline requires five experiment folds")
    outputs = []
    oof_labels: dict[str, int] = {}
    oof_probabilities: dict[str, np.ndarray] = {}
    seed = int(config["project"]["seed"])
    for fold_id, fold in enumerate(folds):
        indexed = data.genotypes.set_index("SampleID")
        train = indexed.loc[fold["train"]].reset_index()
        val = indexed.loc[fold["val"]].reset_index()
        test = indexed.loc[fold["test"]].reset_index()
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000, random_state=seed + fold_id, multi_class="multinomial")),
        ])
        # StandardScaler and LogisticRegression are fit on the three train folds only.
        pipeline.fit(dosage_matrix(train, data.rsid_order), train["Label"].to_numpy(dtype=int))
        # Validation is isolated and may be inspected, but never contributes to fitting or test selection.
        _ = pipeline.predict_proba(dosage_matrix(val, data.rsid_order))
        probabilities = pipeline.predict_proba(dosage_matrix(test, data.rsid_order))
        ordered = np.zeros((len(test), len(data.label_mapping)), dtype=float)
        for source_index, label in enumerate(pipeline.named_steps["logreg"].classes_):
            ordered[:, int(label)] = probabilities[:, source_index]
        metrics, per_class, _ = compute_metrics(test["Label"].to_numpy(dtype=int), ordered, data.label_mapping)
        metrics = {name: metrics[name] for name in METRIC_NAMES}
        outputs.append({
            "experiment_id": fold_id,
            "train_base_fold_ids": [i for i in range(5) if i not in {fold_id, (fold_id + 1) % 5}],
            "val_base_fold_id": (fold_id + 1) % 5,
            "test_base_fold_id": fold_id,
            "metrics": metrics,
            "per_class": per_class.to_dict(orient="records"),
        })
        for sample_id, label, probability in zip(test["SampleID"], test["Label"], ordered, strict=True):
            if str(sample_id) in oof_labels:
                raise ValueError(f"duplicate baseline OOF prediction for {sample_id}")
            oof_labels[str(sample_id)] = int(label); oof_probabilities[str(sample_id)] = probability
    if set(oof_labels) != set(data.genotypes["SampleID"].astype(str)):
        raise ValueError("baseline OOF predictions do not cover every sample exactly once")
    ordered_ids = data.genotypes["SampleID"].astype(str).tolist()
    oof_y = np.asarray([oof_labels[sample_id] for sample_id in ordered_ids]); oof_p = np.asarray([oof_probabilities[sample_id] for sample_id in ordered_ids])
    pooled, pooled_per_class, _ = compute_metrics(oof_y, oof_p, data.label_mapping)
    pooled = {name: pooled[name] for name in METRIC_NAMES}
    summary = {name: {"mean": float(np.mean([fold["metrics"][name] for fold in outputs])), "std": float(np.std([fold["metrics"][name] for fold in outputs], ddof=0))} for name in METRIC_NAMES}
    cv_manifest = resolve_path(config, config["artifacts"]["cv_dir"]) / "folds.json"
    result = {
        "schema_version": "3.0", "pipeline_name": "logreg_baseline", "completed": True,
        "x": int(config["data"]["added_aims_per_autosome"]), "n_aim": data.n_aim,
        "fold_metrics": outputs, "fold_summary": summary, "oof_pooled_metrics": pooled,
        "oof_per_class": pooled_per_class.to_dict(orient="records"),
        "provenance": {"created_at_utc": utc_now_iso(), "seed": seed, "genotype_sha256": sha256_file(resolve_path(config, config["data"]["genotype_csv"])), "cv_manifest_sha256": sha256_file(cv_manifest), "model": {"class": "LogisticRegression", "preprocessing": "StandardScaler", "max_iter": 1000}, "runtime": runtime_metadata(seed)},
    }
    output_path = resolve_path(config, config["artifacts"]["baseline_dir"]) / "logreg_5cv_metrics.json"
    write_json(output_path, result)
    return result
