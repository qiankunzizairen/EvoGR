from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray, label_mapping: dict[int, str]) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    labels = sorted(label_mapping)
    y_pred = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    y_bin = label_binarize(y_true, classes=labels)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            per_auc = roc_auc_score(y_bin, probabilities, average=None)
            macro_auc = roc_auc_score(y_bin, probabilities, average="macro")
        except ValueError:
            per_auc = np.full(len(labels), np.nan)
            macro_auc = np.nan
        per_auprc = average_precision_score(y_bin, probabilities, average=None)
        macro_auprc = average_precision_score(y_bin, probabilities, average="macro")
    overall = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro-Precision": float(np.mean(precision)),
        "Macro-Recall": float(np.mean(recall)),
        "Macro-F1": float(np.mean(f1)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "Macro-AUROC": float(macro_auc) if not np.isnan(macro_auc) else None,
        "Macro-AUPRC": float(macro_auprc),
        "warnings": [str(item.message) for item in caught],
    }
    per_class = pd.DataFrame(
        {
            "Label": labels,
            "PopulationName": [label_mapping[label] for label in labels],
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "AUROC_OvR": per_auc,
            "AUPRC_OvR": per_auprc,
            "Support": support,
        }
    )
    return overall, per_class, confusion_matrix(y_true, y_pred, labels=labels)
