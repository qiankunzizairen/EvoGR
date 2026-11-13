#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from aim_evo2.config import load_config
from aim_evo2.data import load_validated_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AIM56 data contracts.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    data = load_validated_data(load_config(args.config))
    counts = data.genotypes["Label"].value_counts().sort_index()
    print(f"Samples: {len(data.genotypes)}")
    print(f"Labels: {sorted(data.label_mapping)}")
    print(f"Loci: {len(data.rsid_order)}")
    print("Label counts:")
    for label, count in counts.items():
        print(f"  {label}\t{data.label_mapping[int(label)]}\t{count}")
    print("Genotypes: all values are phased biallelic GT strings")


if __name__ == "__main__":
    main()
