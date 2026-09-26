#!/usr/bin/env python3
"""Run the formal fixed-full five-fold train/validation/test pipeline."""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from aim_evo2.config import load_config, resolve_path
from aim_evo2.data import load_validated_data
from aim_evo2.fixed_full import run_fixed_full_cv
from aim_evo2.splits import load_outer_folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-full CV: 3 base folds train, 1 validation, 1 test.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    data = load_validated_data(config)
    folds_path = resolve_path(config, config["artifacts"]["cv_dir"]) / "folds.json"
    folds = load_outer_folds(folds_path)
    result = run_fixed_full_cv(config, data, folds)
    print(f"fixed-full CV complete: folds={len(result['fold_metrics'])} n_aim={result['n_aim']} context_length={result['context_length']}")


if __name__ == "__main__":
    main()
