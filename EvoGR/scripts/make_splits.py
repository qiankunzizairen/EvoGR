#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from aim_evo2.config import load_config
from aim_evo2.data import load_validated_data
from aim_evo2.splits import write_outer_folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the persistent five base folds shared by all pipelines.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    data = load_validated_data(config)
    out = Path(config["_root"]) / config["artifacts"]["cv_dir"] / "folds.json"
    write_outer_folds(data, out, 5, int(config["cv"]["random_state"]))
    print(f"Base folds: {out}")


if __name__ == "__main__":
    main()
