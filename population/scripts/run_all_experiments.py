#!/usr/bin/env python3
"""唯一正式 orchestrator：preflight/cache -> baseline -> three CVs -> result validation."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from aim_evo2.config import load_config, resolve_path
from aim_evo2.data import load_validated_data
from aim_evo2.manifests import assert_manifest_current, embedding_manifest_contract, representation_manifest_contract
from aim_evo2.result_validation import validate_results_directory
from aim_evo2.splits import load_outer_folds


def _run(script: str, config: str) -> None:
    subprocess.run([sys.executable, script, "--config", config], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preflight, cache preparation, baseline, all three CV pipelines, and result validation.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    config = load_config(args.config); root = Path(config["_root"])
    _run("scripts/check_deployment_assets.py", args.config)
    sequence_manifest = resolve_path(config, config["artifacts"]["sequence_dir"]) / "aim_allele_manifest.tsv"
    embedding = resolve_path(config, config["artifacts"]["embedding_dir"]) / "allele_embeddings.pt"
    representation = resolve_path(config, config["artifacts"]["representation_dir"]) / "representation_manifest.json"
    data = load_validated_data(config)
    cache_ready = False
    if sequence_manifest.is_file() and embedding.is_file() and representation.is_file():
        try:
            embedding_manifest_path = embedding.parent / "embedding_manifest.json"
            assert_manifest_current(embedding_manifest_path, embedding_manifest_contract(config, data), "embedding")
            embedding_manifest = __import__("json").loads(embedding_manifest_path.read_text(encoding="utf-8"))
            assert_manifest_current(representation, representation_manifest_contract(config, data, embedding, int(embedding_manifest["hidden_dim"])), "representation")
            cache_ready = True
        except (OSError, ValueError, KeyError):
            cache_ready = False
    if not cache_ready:
        _run("scripts/build_aim_sequences.py", args.config)
        _run("scripts/extract_evo2_embeddings.py", args.config)
        _run("scripts/precompute_sample_representations.py", args.config)
    folds = load_outer_folds(resolve_path(config, config["artifacts"]["cv_dir"]) / "folds.json")
    from aim_evo2.baseline import run_logreg_5cv
    run_logreg_5cv(config, data, folds)
    _run("scripts/run_fixed_full_cv.py", args.config)
    _run("scripts/run_variable_snp_cv.py", args.config)
    _run("scripts/run_variable_context_cv.py", args.config)
    statuses = validate_results_directory(root / "results")
    print("all experiments complete: " + ", ".join(f"{key}={value}" for key, value in statuses.items()))


if __name__ == "__main__":
    main()
