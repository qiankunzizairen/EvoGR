#!/usr/bin/env python3
"""Precompute diploid representations for every configured context length."""
from __future__ import annotations
import argparse
import numpy as np
import _bootstrap  # noqa: F401
from aim_evo2.config import all_context_lengths, load_config, resolve_path
from aim_evo2.data import load_validated_data
from aim_evo2.manifests import assert_manifest_current, embedding_manifest_contract, representation_manifest_contract
from aim_evo2.representations import load_delta_embeddings_by_context
from aim_evo2.utils import utc_now_iso, write_json

def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute diploid representations for all configured context lengths.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    data = load_validated_data(config)
    embedding_dir = resolve_path(config, config["artifacts"]["embedding_dir"])
    embedding_path = embedding_dir / "allele_embeddings.pt"
    assert_manifest_current(embedding_dir / "embedding_manifest.json", embedding_manifest_contract(config, data), "embedding")
    lengths = all_context_lengths(config)
    deltas = load_delta_embeddings_by_context(str(embedding_path), int(config["evo2"]["active_pool_radius"]), data.rsid_order, lengths)
    raw = data.genotypes[data.rsid_order].to_numpy(dtype=str)
    allele_a = np.fromiter((value[0] for value in raw.ravel()), dtype=np.int8, count=raw.size).reshape(raw.shape)
    allele_b = np.fromiter((value[2] for value in raw.ravel()), dtype=np.int8, count=raw.size).reshape(raw.shape)
    dosage = (allele_a + allele_b).astype(np.float32)
    heterozygous = (allele_a != allele_b).astype(np.float32)
    output_dir = resolve_path(config, config["artifacts"]["representation_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    hidden_dim = None
    for length in lengths:
        delta = deltas[length]
        hidden_dim = int(delta.shape[1])
        target = output_dir / f"representations_L{length}.npy"
        array = np.lib.format.open_memmap(target, mode="w+", dtype="float16", shape=(len(dosage), data.n_aim, hidden_dim * 2))
        delta_np = delta.numpy()
        for start in range(0, len(dosage), 32):
            stop = min(start + 32, len(dosage))
            mean = dosage[start:stop, :, None] * 0.5 * delta_np[None, :, :]
            het = heterozygous[start:stop, :, None] * np.abs(delta_np[None, :, :])
            array[start:stop] = np.concatenate([mean, het], axis=2).astype("float16")
        array.flush()
        del array
    assert hidden_dim is not None
    manifest = representation_manifest_contract(config, data, embedding_path, hidden_dim)
    manifest["created_at_utc"] = utc_now_iso()
    write_json(output_dir / "representation_manifest.json", manifest)
    print(f"Representations: {output_dir} ({len(lengths)} context lengths)")

if __name__ == "__main__":
    main()
