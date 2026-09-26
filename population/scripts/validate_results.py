#!/usr/bin/env python3
"""Read-only validation of the three formal result JSON files."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import _bootstrap  # noqa: F401
from aim_evo2.config import load_config, resolve_path
from aim_evo2.result_validation import validate_results_directory

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate formal fixed-full and variable pipeline result schemas.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    try:
        statuses = validate_results_directory(resolve_path(load_config(args.config), "results"))
    except (OSError, ValueError) as exc:
        print(f"result validation FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print("result validation OK: " + ", ".join(f"{key}={value}" for key, value in statuses.items()))

if __name__ == "__main__":
    main()
