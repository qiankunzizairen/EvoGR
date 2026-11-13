#!/usr/bin/env python3
"""Run the raw-genotype dosage Logistic Regression 5-fold baseline."""
from __future__ import annotations
import argparse
import _bootstrap  # noqa: F401
from aim_evo2.baseline import run_logreg_5cv
from aim_evo2.config import load_config, resolve_path
from aim_evo2.data import load_validated_data
from aim_evo2.splits import load_outer_folds

def main() -> None:
    parser = argparse.ArgumentParser(description="Run dosage-only multinomial Logistic Regression 5-fold CV.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args(); config = load_config(args.config); data = load_validated_data(config)
    folds = load_outer_folds(resolve_path(config, config["artifacts"]["cv_dir"]) / "folds.json")
    result = run_logreg_5cv(config, data, folds)
    print(f"baseline complete: folds={len(result['fold_metrics'])} n_aim={result['n_aim']}")

if __name__ == "__main__":
    main()
