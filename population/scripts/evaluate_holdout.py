#!/usr/bin/env python3
"""Retired compatibility entry point; formal evaluation uses five-fold CV."""
import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Retired holdout evaluator (use run_cv.py).")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.parse_args()
    raise SystemExit("Holdout evaluation is retired; use the five-fold CV scripts.")

if __name__ == "__main__":
    main()
