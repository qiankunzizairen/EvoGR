from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .fixed_full import METRIC_NAMES


FORMAL_RESULT_NAMES = ("fixed_full", "variable_snp", "variable_context")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_result_payload(payload: dict[str, Any], pipeline: str) -> None:
    _require(payload.get("schema_version") == "3.0", f"{pipeline}: schema_version must be 3.0")
    _require(payload.get("pipeline_name") == pipeline, f"{pipeline}: pipeline_name mismatch")
    _require(isinstance(payload.get("x"), int) and isinstance(payload.get("n_aim"), int), f"{pipeline}: missing x/n_aim")
    _require(isinstance(payload.get("provenance"), dict) and payload["provenance"].get("created_at_utc"), f"{pipeline}: missing provenance/created time")
    if payload.get("completed") is False:
        _require(payload.get("status") == "pending_linux", f"{pipeline}: incomplete result must be pending_linux")
        return
    _require(payload.get("completed") is True, f"{pipeline}: completed must be true or false")
    folds = payload.get("folds") or payload.get("fold_metrics")
    _require(isinstance(folds, list) and len(folds) == 5, f"{pipeline}: expected five experiment folds")
    for index, fold in enumerate(folds):
        _require(isinstance(fold, dict), f"{pipeline}: fold {index} is not a mapping")
        _require({"train_base_fold_ids", "val_base_fold_id", "test_base_fold_id"}.issubset(fold), f"{pipeline}: fold {index} mapping missing")
        _require(set(fold["train_base_fold_ids"]) == {i for i in range(5) if i not in {index, (index + 1) % 5}}, f"{pipeline}: fold {index} train mapping mismatch")
        _require(fold["val_base_fold_id"] == (index + 1) % 5 and fold["test_base_fold_id"] == index, f"{pipeline}: fold {index} val/test mapping mismatch")
    provenance = payload["provenance"]
    for key in ("genotype_sha256", "embedding_manifest_sha256", "representation_manifest_sha256", "cv_manifest_sha256"):
        _require(provenance.get(key), f"{pipeline}: provenance missing {key}")
    if pipeline == "fixed_full":
        groups = [fold.get("metrics") for fold in folds]
        summary = payload.get("fold_summary")
    elif pipeline == "variable_snp":
        groups = [entry.get("metrics") for fold in folds for entry in fold.get("metrics_by_k", {}).values()]
        summary = payload.get("fold_metrics_by_k")
        _require(payload.get("flank_bp") == 50 and payload.get("context_length") == 101, "variable_snp: fixed flank/L mismatch")
        _require(payload.get("k_values") and payload.get("metrics_by_k"), "variable_snp: missing K results")
    else:
        groups = [entry.get("metrics") for fold in folds for entry in fold.get("metrics_by_context_length", {}).values()]
        summary = payload.get("fold_metrics_by_context_length")
        _require(payload.get("train_n") == [10, 50, 100, 500, 1000] and payload.get("eval_n") == [10, 50, 100, 500, 1000], "variable_context: n grid mismatch")
        _require(payload.get("metrics_by_context_length"), "variable_context: missing context results")
    _require(all(isinstance(group, dict) and set(group) == set(METRIC_NAMES) and all(group[name] is not None for name in METRIC_NAMES) for group in groups), f"{pipeline}: every fold/group must contain seven non-null metrics")
    _require(isinstance(summary, dict) and summary, f"{pipeline}: missing fold mean/std summary")
    if pipeline == "fixed_full":
        _require(set(summary) == set(METRIC_NAMES) and all(isinstance(summary[name], dict) and set(summary[name]) == {"mean", "std"} for name in METRIC_NAMES), f"{pipeline}: missing mean/std metric")
    else:
        for group_summary in summary.values():
            _require(isinstance(group_summary, dict) and all(isinstance(group_summary.get(name), dict) and set(group_summary[name]) == {"mean", "std"} for name in METRIC_NAMES), f"{pipeline}: missing mean/std metric")
    _require(payload.get("oof_pooled_metrics"), f"{pipeline}: missing OOF pooled metrics")
    _require(payload.get("oof_per_class"), f"{pipeline}: missing OOF per-class metrics")


def validate_results_directory(results_dir: Path) -> dict[str, str]:
    expected = {f"{pipeline}/test_metrics.json" for pipeline in FORMAL_RESULT_NAMES}
    actual = {str(path.relative_to(results_dir)) for path in results_dir.rglob("*") if path.is_file() and path.name != ".DS_Store"}
    if actual != expected:
        raise ValueError(f"results directory mismatch: expected {sorted(expected)}, found {sorted(actual)}")
    statuses = {}
    for pipeline in FORMAL_RESULT_NAMES:
        path = results_dir / pipeline / "test_metrics.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{pipeline}: unreadable JSON: {exc}") from exc
        validate_result_payload(payload, pipeline)
        statuses[pipeline] = "pending" if payload.get("completed") is False else "complete"
    return statuses
