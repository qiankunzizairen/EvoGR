#!/usr/bin/env python3
"""Train the read-based population classifier.

Usage: run_reads.py READ_DIRECTORY LABEL_MAPPING
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from aim_evo2.read_data import load_read_samples
from aim_evo2.read_training import embed_reads, train_read_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a population classifier from variable-length single-stranded reads.")
    parser.add_argument("read_directory", type=Path, help="folder containing one .txt file per sample")
    parser.add_argument("label_mapping", type=Path, help="two-column FILE,LABEL mapping")
    parser.add_argument("--num-classes", type=int, default=None, help="total number of population classes")
    parser.add_argument("--checkpoint", type=Path, required=True, help="local Evo2 checkpoint")
    parser.add_argument("--model-name", default="evo2_7b_base")
    parser.add_argument("--layer-name", required=True, help="Evo2 block layer used for read embeddings")
    parser.add_argument("--max-read-length", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("population_read_head.pt"))
    args = parser.parse_args()
    if not args.read_directory.is_dir():
        parser.error(f"read directory does not exist: {args.read_directory}")
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs and batch-size must be positive and learning-rate must be greater than zero")
    samples, label_to_id = load_read_samples(args.read_directory, args.label_mapping, args.num_classes)
    class_count = args.num_classes or len(label_to_id)
    embeddings, hidden_dim = embed_reads(samples, args.checkpoint, args.model_name, args.layer_name, args.max_read_length)
    model = train_read_classifier(embeddings, samples, class_count, args.epochs, args.batch_size, args.learning_rate, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch = __import__("torch")
    torch.save({"model_state_dict": model.state_dict(), "hidden_dim": hidden_dim, "num_classes": class_count, "label_to_id": label_to_id}, args.output)
    print(f"saved read classifier: {args.output}")


if __name__ == "__main__":
    main()
