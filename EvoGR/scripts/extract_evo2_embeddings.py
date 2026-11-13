#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from aim_evo2.config import load_config
from aim_evo2.evo2_embed import extract_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen Evo2 allele embeddings on Linux.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    embedding_path, manifest_path = extract_embeddings(load_config(args.config))
    print(f"Embeddings: {embedding_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
