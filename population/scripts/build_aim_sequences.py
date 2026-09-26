#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from aim_evo2.config import load_config
from aim_evo2.sequences import build_sequences_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build configured odd-length REF/ALT SNP sequences.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    fasta_path, manifest_path = build_sequences_from_config(load_config(args.config))
    print(f"FASTA: {fasta_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
