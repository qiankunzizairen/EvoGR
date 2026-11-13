from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .data import load_validated_data
from .reference import IndexedFasta
from .config import all_context_lengths, all_flank_bp, fixed_flank_bp
from .utils import sha256_file, utc_now_iso, write_json


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def build_locus_sequences(
    fasta: IndexedFasta,
    alleles: pd.DataFrame,
    context_length: int,
    center_index_0based: int | None = None,
) -> tuple[list[tuple[str, str]], list[dict[str, str | int]]]:
    if context_length < 1:
        raise ValueError(f"context_length must be positive, found {context_length}")
    expected_center = context_length // 2
    if center_index_0based is None:
        center_index_0based = expected_center
    if center_index_0based != expected_center:
        raise ValueError(f"center index must be n={expected_center} for L={context_length}")
    records: list[tuple[str, str]] = []
    manifest: list[dict[str, str | int]] = []
    for _, row in alleles.iterrows():
        rsid = str(row["rsID"])
        chrom = str(row["CHROM"])
        pos = int(row["POS"])
        ref = str(row["REF"]).upper()
        alt = str(row["ALT"]).upper()
        variant_pos0 = pos - 1
        start0 = variant_pos0 - center_index_0based
        end0 = start0 + context_length
        ref_sequence = fasta.fetch(chrom, start0, end0)
        if len(ref_sequence) != context_length:
            raise ValueError(f"{rsid}: expected {context_length} bp, found {len(ref_sequence)}")
        if ref_sequence[center_index_0based] != ref:
            raise ValueError(
                f"{rsid}: reference mismatch at {chrom}:{pos}; "
                f"FASTA={ref_sequence[center_index_0based]} metadata_REF={ref}"
            )
        alt_sequence = ref_sequence[:center_index_0based] + alt + ref_sequence[center_index_0based + 1 :]
        for allele_index, allele_base, sequence in [
            (0, ref, ref_sequence),
            (1, alt, alt_sequence),
        ]:
            seq_id = f"{rsid}_allele{allele_index}_{allele_base}"
            records.append((seq_id, sequence))
            manifest.append(
                {
                    "seq_id": seq_id,
                    "rsID": rsid,
                    "CHROM": chrom,
                    "POS": pos,
                    "REF": ref,
                    "ALT": alt,
                    "allele_index": allele_index,
                    "allele_base": allele_base,
                    "window_start_1based": start0 + 1,
                    "window_end_1based": end0,
                    "center_index_0based": center_index_0based,
                    "sequence_length": len(sequence),
                    "sequence_sha256": sequence_sha256(sequence),
                }
            )
    return records, manifest


def build_locus_sequences_for_lengths(fasta: IndexedFasta, alleles: pd.DataFrame, context_lengths: list[int]) -> tuple[list[tuple[str, str]], list[dict[str, str | int]]]:
    records, manifest = [], []
    for length in context_lengths:
        center = length // 2
        current, current_manifest = build_locus_sequences(fasta, alleles, length, center)
        for seq_id, sequence in current:
            records.append((f"L{length}_{seq_id}", sequence))
        for row in current_manifest:
            row = dict(row)
            row["seq_id"] = f"L{length}_{row['seq_id']}"
            row["context_length"] = length
            manifest.append(row)
    return records, manifest


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        for seq_id, sequence in records:
            handle.write(f">{seq_id}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def build_sequences_from_config(config: dict) -> tuple[Path, Path]:
    data = load_validated_data(config)
    fasta = IndexedFasta(
        Path(config["_root"]) / config["data"]["reference_fasta"],
        Path(config["_root"]) / config["data"]["reference_fai"],
    )
    lengths = all_context_lengths(config)
    records, manifest = build_locus_sequences_for_lengths(fasta, data.alleles, lengths)
    output_dir = Path(config["_root"]) / config["artifacts"]["sequence_dir"]
    fasta_path = output_dir / "aim_alleles.fasta"
    manifest_path = output_dir / "aim_allele_manifest.tsv"
    write_fasta(records, fasta_path)
    pd.DataFrame(manifest).to_csv(manifest_path, sep="\t", index=False)
    write_json(
        output_dir / "aim_sequence_build_manifest.json",
        {
            "created_at_utc": utc_now_iso(),
            "fasta_path": str(fasta_path),
            "fasta_sha256": sha256_file(fasta_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "sequence_count": len(records),
            "flank_bp": all_flank_bp(config),
            "fixed_flank_bp": fixed_flank_bp(config),
            "context_lengths": lengths,
            "n_aim": data.n_aim,
            "x": int(config["data"]["added_aims_per_autosome"]),
            "rsid_order": data.rsid_order,
        },
    )
    return fasta_path, manifest_path
