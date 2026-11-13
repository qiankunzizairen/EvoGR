#!/usr/bin/env python3
"""Run variable-context CV: fixed K=N_AIM, sampled n/L."""
from __future__ import annotations
import argparse
import _bootstrap  # noqa: F401
from aim_evo2.config import load_config, resolve_path
from aim_evo2.data import load_validated_data
from aim_evo2.splits import load_outer_folds
from aim_evo2.variable_pipelines import run_variable_context_cv

def main() -> None:
    parser = argparse.ArgumentParser(description="Run variable-context CV with deterministic per-context evaluation.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args(); config = load_config(args.config); data = load_validated_data(config)
    folds = load_outer_folds(resolve_path(config, config["artifacts"]["cv_dir"]) / "folds.json")
    result = run_variable_context_cv(config, data, folds)
    print(f"variable-context CV complete: folds={len(result['folds'])} L={result['eval_context_lengths']} n_aim={result['n_aim']}")

if __name__ == "__main__":
    main()
