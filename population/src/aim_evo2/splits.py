from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from .data import ValidatedData
from .utils import sha256_file, write_json
import json


def make_holdout_split(
    data: ValidatedData,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_fraction, random_state=seed)
    labels = data.genotypes["Label"].to_numpy()
    train_idx, test_idx = next(splitter.split(data.genotypes, labels))
    train = data.genotypes.iloc[train_idx][["SampleID", "Label"]].sort_values("SampleID").reset_index(drop=True)
    test = data.genotypes.iloc[test_idx][["SampleID", "Label"]].sort_values("SampleID").reset_index(drop=True)
    if set(train["SampleID"]) & set(test["SampleID"]):
        raise ValueError("Train/test SampleID overlap")
    summary_rows = []
    for label in sorted(data.label_mapping):
        total = int((data.genotypes["Label"] == label).sum())
        train_count = int((train["Label"] == label).sum())
        test_count = int((test["Label"] == label).sum())
        if train_count == 0 or test_count == 0:
            raise ValueError(f"Label {label} missing from train or test")
        summary_rows.append(
            {
                "Label": label,
                "PopulationName": data.label_mapping[label],
                "Total": total,
                "Train": train_count,
                "Test": test_count,
                "TrainFraction": train_count / total,
                "TestFraction": test_count / total,
            }
        )
    return train, test, pd.DataFrame(summary_rows)


def make_inner_validation_split(train: pd.DataFrame, validation_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    inner_train_idx, inner_val_idx = next(splitter.split(train, train["Label"]))
    return (
        train.iloc[inner_train_idx]["SampleID"].tolist(),
        train.iloc[inner_val_idx]["SampleID"].tolist(),
    )


def make_cv_folds(data: ValidatedData, n_splits: int = 5, seed: int = 42) -> list[tuple[list[str], list[str]]]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in splitter.split(data.genotypes, data.genotypes["Label"]):
        folds.append(
            (
                data.genotypes.iloc[train_idx]["SampleID"].tolist(),
                data.genotypes.iloc[val_idx]["SampleID"].tolist(),
            )
        )
    return folds


def make_base_folds(data: ValidatedData, n_splits: int = 5, seed: int = 42) -> list[list[str]]:
    """Create the persistent, disjoint validation partitions shared by all pipelines."""
    if n_splits != 5:
        raise ValueError("Formal evaluation requires exactly five base folds")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return [data.genotypes.iloc[test_idx]["SampleID"].tolist() for _, test_idx in splitter.split(data.genotypes, data.genotypes["Label"])]


def make_experiment_folds(base_folds: list[list[str]]) -> list[dict[str, list[str]]]:
    if len(base_folds) != 5 or any(not fold for fold in base_folds):
        raise ValueError("exactly five non-empty base folds are required")
    all_ids = [sample_id for fold in base_folds for sample_id in fold]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("base folds overlap")
    result = []
    for experiment_id in range(5):
        validation_id = (experiment_id + 1) % 5
        test_id = experiment_id
        train_ids = [sample_id for fold_id, fold in enumerate(base_folds) if fold_id not in {validation_id, test_id} for sample_id in fold]
        result.append({"experiment_id": experiment_id, "train": train_ids, "val": list(base_folds[validation_id]), "test": list(base_folds[test_id])})
    return result


def make_outer_folds(data: ValidatedData, n_splits: int = 5, seed: int = 42) -> list[dict[str, list[str]]]:
    return make_experiment_folds(make_base_folds(data, n_splits, seed))


def write_outer_folds(data: ValidatedData, output_path: Path, n_splits: int = 5, seed: int = 42) -> list[dict[str, list[str]]]:
    folds = make_outer_folds(data, n_splits, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "2.0", "seed": seed, "n_splits": n_splits,
               "base_folds": [f["test"] for f in folds], "experiment_folds": folds,
               "n_aim": data.n_aim, "x": int(data.n_aim - 56) // 22}
    write_json(output_path, payload)
    return folds


def load_outer_folds(path: Path | str) -> list[dict[str, list[str]]]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds = payload.get("experiment_folds")
    base_folds = payload.get("base_folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError(f"{path}: expected schema_version 2.0 experiment folds")
    if not isinstance(base_folds, list) or len(base_folds) != 5:
        raise ValueError(f"{path}: expected five persisted base_folds")
    for fold in folds:
        if set(fold) != {"experiment_id", "train", "val", "test"}:
            raise ValueError(f"{path}: invalid experiment-fold mapping")
        index = int(fold["experiment_id"])
        if index not in range(5) or set(fold["val"]) != set(base_folds[(index + 1) % 5]) or set(fold["test"]) != set(base_folds[index]):
            raise ValueError(f"{path}: experiment {index} violates fixed val/test mapping")
        expected_train = {sid for base_id, values in enumerate(base_folds) if base_id not in {index, (index + 1) % 5} for sid in values}
        if set(fold["train"]) != expected_train:
            raise ValueError(f"{path}: experiment {index} violates fixed train mapping")
    return folds
