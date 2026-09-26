#!/usr/bin/env python3
"""Run variable-SNP CV: fixed L=101, sampled K."""
from __future__ import annotations
import argparse
import _bootstrap  # noqa: F401
from aim_evo2.config import load_config, resolve_path
from aim_evo2.data import load_validated_data
from aim_evo2.splits import load_outer_folds
from aim_evo2.variable_pipelines import run_variable_snp_cv

def main() -> None:
    parser = argparse.ArgumentParser(description="Run variable-SNP CV with deterministic validation/test subsets.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args(); config = load_config(args.config); data = load_validated_data(config)
    folds = load_outer_folds(resolve_path(config, config["artifacts"]["cv_dir"]) / "folds.json")
    result = run_variable_snp_cv(config, data, folds)
    print(f"variable-SNP CV complete: folds={len(result['folds'])} K={result['k_values']} n_aim={result['n_aim']}")

if __name__ == "__main__":
    main()
