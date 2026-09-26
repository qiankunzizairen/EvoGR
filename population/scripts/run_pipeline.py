#!/usr/bin/env python3
from __future__ import annotations

import argparse
import _bootstrap  # noqa: F401
from aim_evo2.config import load_config
from aim_evo2.data import load_validated_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data validation, split creation, and sequence checks before model-specific stages.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    data = load_validated_data(config)
    print(f"Foundation validation passed: x={config['data']['added_aims_per_autosome']} n_aim={data.n_aim}")


if __name__ == "__main__":
    main()
