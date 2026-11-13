#!/usr/bin/env python3
"""Run the three reproducible 5-fold experiment pipelines."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import _bootstrap  # noqa: F401
from aim_evo2.config import load_config, resolve_path
from aim_evo2.data import load_validated_data
from aim_evo2.splits import load_outer_folds, write_outer_folds

def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-full, variable-SNP, and variable-context 5-fold CV")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--pipeline", choices=["all", "fixed_full", "variable_snp", "variable_context"], default="all")
    parser.add_argument("--folds", type=Path, default=None, help="Persistent outer-fold JSON")
    args = parser.parse_args()
    config = load_config(args.config)
    data = load_validated_data(config)
    folds_path = args.folds or (resolve_path(config, config["artifacts"]["cv_dir"]) / "folds.json")
    folds = load_outer_folds(folds_path) if folds_path.exists() else write_outer_folds(data, folds_path, 5, int(config["cv"]["random_state"]))
    print(json.dumps({"n_aim": data.n_aim, "x": config["data"]["added_aims_per_autosome"], "folds": len(folds), "pipeline": args.pipeline}, indent=2))
    try:
        if args.pipeline == "fixed_full":
            from aim_evo2.fixed_full import run_fixed_full_cv
            run_fixed_full_cv(config, data, folds)
        elif args.pipeline == "variable_snp":
            from aim_evo2.variable_pipelines import run_variable_snp_cv
            run_variable_snp_cv(config, data, folds)
        elif args.pipeline == "variable_context":
            from aim_evo2.variable_pipelines import run_variable_context_cv
            run_variable_context_cv(config, data, folds)
        else:
            raise SystemExit("--pipeline all is retired; run each named pipeline explicitly")
    except ModuleNotFoundError:
        raise SystemExit("CV runner module is unavailable; install the project package first")

if __name__ == "__main__":
    main()
