#!/usr/bin/env python3
"""Build the unrelated-sample AIM genotype matrix and VCF allele manifest.

The 2504 high-coverage sequence index defines the sample projection. Population
metadata comes from the 3202-sample pedigree table. rsIDs are mapped onto VCF
records by normalized CHROM and POS; REF and ALT are always copied from the VCF.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

DEFAULT_PEDIGREE = DATA_DIR / "20130606_g1k_3202_samples_ped_population.txt"
DEFAULT_SEQUENCE_INDEX = DATA_DIR / "1000G_2504_high_coverage.sequence.index"
DEFAULT_VCF = DATA_DIR / "aims56_1000g.vcf"
DEFAULT_TARGETS = DATA_DIR / "aims56_grch38.csv"
DEFAULT_CHN_VCF = DATA_DIR / "CHN100K_aims_1000g.vcf"
DEFAULT_CHN_TARGETS_DIR = DATA_DIR / "CHN100K_aims"

EXPECTED_UNRELATED_SAMPLES = 2504
EXPECTED_SNPS = 56


def normalize_chrom(value: str) -> str:
    """Normalize only the optional VCF-style ``chr`` prefix."""
    value = value.strip()
    if value.lower().startswith("chr"):
        value = value[3:]
    return value.upper()


def require_columns(columns: Iterable[str], required: set[str], source: Path) -> None:
    missing = required - set(columns)
    if missing:
        raise ValueError(f"{source}: missing required columns: {sorted(missing)}")


def read_population_table(path: Path) -> dict[str, tuple[str, str]]:
    """Return SampleID -> (Superpopulation, Population)."""
    with path.open(encoding="utf-8") as handle:
        header_line = handle.readline()
        if not header_line:
            raise ValueError(f"{path}: empty population table")
        header = header_line.split()
        require_columns(
            header,
            {"SampleID", "Population", "Superpopulation"},
            path,
        )

        samples: dict[str, tuple[str, str]] = {}
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            values = line.split()
            if len(values) != len(header):
                raise ValueError(
                    f"{path}:{line_number}: expected {len(header)} fields, "
                    f"found {len(values)}"
                )
            row = dict(zip(header, values))
            sample_id = row["SampleID"].strip()
            if sample_id in samples:
                raise ValueError(f"{path}:{line_number}: duplicate SampleID {sample_id}")
            samples[sample_id] = (
                row["Superpopulation"].strip(),
                row["Population"].strip(),
            )

    return samples


def read_sequence_index(path: Path) -> tuple[list[str], dict[str, str]]:
    """Return ordered SAMPLE_NAME values and their index populations."""
    header: list[str] | None = None
    sample_ids: list[str] = []
    populations: dict[str, str] = {}

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("##") or not line.strip():
                continue
            if line.startswith("#"):
                header = [value.strip() for value in line[1:].rstrip("\n").split("\t")]
                require_columns(header, {"SAMPLE_NAME", "POPULATION"}, path)
                continue
            if header is None:
                raise ValueError(f"{path}:{line_number}: data appears before the header")

            values = [value.strip() for value in line.rstrip("\n").split("\t")]
            if len(values) != len(header):
                raise ValueError(
                    f"{path}:{line_number}: expected {len(header)} fields, "
                    f"found {len(values)}"
                )
            row = dict(zip(header, values))
            sample_id = row["SAMPLE_NAME"]
            if not sample_id:
                raise ValueError(f"{path}:{line_number}: empty SAMPLE_NAME")
            if sample_id in populations:
                raise ValueError(f"{path}:{line_number}: duplicate SAMPLE_NAME {sample_id}")
            sample_ids.append(sample_id)
            populations[sample_id] = row["POPULATION"]

    if header is None:
        raise ValueError(f"{path}: header not found")
    return sample_ids, populations


def read_rsid_map(path: Path) -> dict[tuple[str, int], str]:
    """Read only rsID/CHROM/POS; database REF/ALT are deliberately ignored."""
    mapping: dict[tuple[str, int], str] = {}
    seen_rsids: set[str] = set()

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV header not found")
        require_columns(reader.fieldnames, {"rsID", "CHROM", "POS"}, path)

        for line_number, row in enumerate(reader, start=2):
            rsid = (row["rsID"] or "").strip()
            chrom = normalize_chrom(row["CHROM"] or "")
            try:
                pos = int((row["POS"] or "").strip())
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid POS {row['POS']!r}") from exc
            if not rsid or not chrom:
                raise ValueError(f"{path}:{line_number}: empty rsID or CHROM")
            key = (chrom, pos)
            if key in mapping:
                raise ValueError(f"{path}:{line_number}: duplicate locus {chrom}:{pos}")
            if rsid in seen_rsids:
                raise ValueError(f"{path}:{line_number}: duplicate rsID {rsid}")
            mapping[key] = rsid
            seen_rsids.add(rsid)

    return mapping


def read_chn100k_rsid_map(directory: Path, top_n: int) -> dict[tuple[str, int], str]:
    """Read the top ``In_bits`` SNPs from each chromosome TSV."""
    mapping: dict[tuple[str, int], str] = {}
    seen_rsids: set[str] = set()
    for chromosome in range(1, 23):
        path = directory / f"chr{chromosome}_top200.tsv"
        if not path.exists():
            raise ValueError(f"{path}: CHN100K chromosome table not found")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError(f"{path}: TSV header not found")
            require_columns(reader.fieldnames, {"rsID", "CHROM", "POS", "In_bits"}, path)
            rows = list(reader)
        try:
            rows.sort(key=lambda row: float((row["In_bits"] or "").strip()), reverse=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid In_bits value") from exc
        if len(rows) < top_n:
            raise ValueError(f"{path}: expected at least {top_n} SNP rows, found {len(rows)}")
        for row in rows[:top_n]:
            rsid = (row["rsID"] or "").strip()
            chrom = normalize_chrom(row["CHROM"] or "")
            try:
                pos = int((row["POS"] or "").strip())
            except ValueError as exc:
                raise ValueError(f"{path}: invalid POS {row['POS']!r}") from exc
            if not rsid or not chrom:
                raise ValueError(f"{path}: empty rsID or CHROM")
            key = (chrom, pos)
            if key in mapping:
                raise ValueError(f"{path}: duplicate locus {chrom}:{pos}")
            if rsid in seen_rsids:
                raise ValueError(f"{path}: duplicate rsID {rsid}")
            mapping[key] = rsid
            seen_rsids.add(rsid)
    return mapping


def read_vcf(
    path: Path,
    rsid_by_locus: dict[tuple[str, int], str],
    projected_samples: list[str],
    *,
    allow_extra_records: bool = False,
) -> tuple[list[str], list[list[str]], list[dict[str, str]]]:
    """Read projected GT values and VCF-derived allele records."""
    sample_columns: list[str] | None = None
    projected_indices: list[int] | None = None
    rsids: list[str] = []
    genotypes_by_sample = [[] for _ in projected_samples]
    alleles: list[dict[str, str]] = []
    seen_loci: set[tuple[str, int]] = set()
    seen_rsids: set[str] = set()

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                columns = line.rstrip("\n").split("\t")
                if len(columns) < 10:
                    raise ValueError(f"{path}:{line_number}: VCF has no sample columns")
                sample_columns = [sample.strip() for sample in columns[9:]]
                if len(sample_columns) != len(set(sample_columns)):
                    raise ValueError(f"{path}:{line_number}: duplicate VCF sample IDs")
                vcf_index = {sample: index for index, sample in enumerate(sample_columns)}
                missing = [sample for sample in projected_samples if sample not in vcf_index]
                if missing:
                    raise ValueError(
                        f"{path}: {len(missing)} projected samples are absent from the VCF: "
                        f"{missing[:10]}"
                    )
                projected_indices = [vcf_index[sample] for sample in projected_samples]
                continue
            if line.startswith("#") or not line.strip():
                continue
            if sample_columns is None or projected_indices is None:
                raise ValueError(f"{path}:{line_number}: VCF record appears before #CHROM")

            fields = line.rstrip("\n").split("\t")
            expected_fields = 9 + len(sample_columns)
            if len(fields) != expected_fields:
                raise ValueError(
                    f"{path}:{line_number}: expected {expected_fields} fields, "
                    f"found {len(fields)}"
                )

            chrom = normalize_chrom(fields[0])
            try:
                pos = int(fields[1])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid POS {fields[1]!r}") from exc
            locus = (chrom, pos)
            if locus not in rsid_by_locus:
                if allow_extra_records:
                    continue
                raise ValueError(
                    f"{path}:{line_number}: locus {chrom}:{pos} is absent from the rsID map"
                )
            # CHN100K extraction can contain an adjacent record whose POS is
            # duplicated but whose ID encodes a different source coordinate;
            # retain the record whose ID agrees with the requested locus.
            if allow_extra_records and not fields[2].startswith(f"{chrom}:{pos}:"):
                continue
            if locus in seen_loci:
                raise ValueError(f"{path}:{line_number}: duplicate locus {chrom}:{pos}")
            seen_loci.add(locus)
            rsid = rsid_by_locus[locus]
            if rsid in seen_rsids:
                raise ValueError(f"{path}:{line_number}: rsID {rsid} maps more than once")
            seen_rsids.add(rsid)

            ref = fields[3]
            alt = fields[4]
            if not ref or ref == "." or not alt or alt == ".":
                raise ValueError(f"{path}:{line_number}: missing REF or ALT for {rsid}")

            format_fields = fields[8].split(":")
            if "GT" not in format_fields:
                raise ValueError(f"{path}:{line_number}: FORMAT has no GT for {rsid}")
            gt_index = format_fields.index("GT")
            sample_fields = fields[9:]
            for output_index, vcf_sample_index in enumerate(projected_indices):
                values = sample_fields[vcf_sample_index].split(":")
                if gt_index >= len(values):
                    raise ValueError(
                        f"{path}:{line_number}: missing GT field for "
                        f"{projected_samples[output_index]} at {rsid}"
                    )
                genotypes_by_sample[output_index].append(values[gt_index])

            rsids.append(rsid)
            alleles.append(
                {
                    "rsID": rsid,
                    "CHROM": chrom,
                    "POS": str(pos),
                    "REF": ref,
                    "ALT": alt,
                }
            )

    if sample_columns is None:
        raise ValueError(f"{path}: #CHROM header not found")

    missing_loci = set(rsid_by_locus) - seen_loci
    if missing_loci:
        preview = [f"{chrom}:{pos}" for chrom, pos in sorted(missing_loci)[:10]]
        raise ValueError(
            f"{path}: {len(missing_loci)} target loci are missing from the VCF: {preview}"
        )
    return rsids, genotypes_by_sample, alleles


def write_csv_atomic(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pedigree", type=Path, default=DEFAULT_PEDIGREE)
    parser.add_argument("--sequence-index", type=Path, default=DEFAULT_SEQUENCE_INDEX)
    parser.add_argument("--vcf", type=Path, default=DEFAULT_VCF)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--chn100k-vcf", type=Path, default=DEFAULT_CHN_VCF)
    parser.add_argument("--chn100k-targets-dir", type=Path, default=DEFAULT_CHN_TARGETS_DIR)
    parser.add_argument(
        "--chn100k-top", type=int, default=10, metavar="N",
        help="number of highest In_bits SNPs per chromosome (1-200; default: 10)",
    )
    parser.add_argument("--genotypes-output", type=Path)
    parser.add_argument("--alleles-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.chn100k_top <= 200:
        raise ValueError("--chn100k-top must be an integer between 1 and 200")
    population_by_sample = read_population_table(args.pedigree)
    projected_samples, index_populations = read_sequence_index(args.sequence_index)

    if len(projected_samples) != EXPECTED_UNRELATED_SAMPLES:
        raise ValueError(
            f"{args.sequence_index}: expected {EXPECTED_UNRELATED_SAMPLES} unique samples, "
            f"found {len(projected_samples)}"
        )

    absent_from_population = [
        sample for sample in projected_samples if sample not in population_by_sample
    ]
    if absent_from_population:
        raise ValueError(
            f"{len(absent_from_population)} projected samples are absent from the population "
            f"table: {absent_from_population[:10]}"
        )

    population_disagreements = [
        sample
        for sample in projected_samples
        if index_populations[sample]
        and index_populations[sample] != population_by_sample[sample][1]
    ]
    if population_disagreements:
        raise ValueError(
            f"Population disagrees between sequence index and population table for "
            f"{len(population_disagreements)} samples: {population_disagreements[:10]}"
        )

    aims_rsid_by_locus = read_rsid_map(args.targets)
    if len(aims_rsid_by_locus) != EXPECTED_SNPS:
        raise ValueError(
            f"{args.targets}: expected {EXPECTED_SNPS} unique SNPs, "
            f"found {len(aims_rsid_by_locus)}"
        )
    chn_rsid_by_locus = read_chn100k_rsid_map(args.chn100k_targets_dir, args.chn100k_top)
    overlap = set(aims_rsid_by_locus) & set(chn_rsid_by_locus)
    if overlap:
        raise ValueError(f"AIM56 and CHN100K targets overlap at {len(overlap)} loci")
    rsid_by_locus = {**aims_rsid_by_locus, **chn_rsid_by_locus}

    aims_rsids, aims_genotypes, aims_alleles = read_vcf(
        args.vcf,
        aims_rsid_by_locus,
        projected_samples,
    )
    chn_rsids, chn_genotypes, chn_alleles = read_vcf(
        args.chn100k_vcf,
        chn_rsid_by_locus,
        projected_samples,
        allow_extra_records=True,
    )
    rsids = aims_rsids + chn_rsids
    genotypes_by_sample = [a + c for a, c in zip(aims_genotypes, chn_genotypes)]
    alleles = aims_alleles + chn_alleles
    if len(aims_rsids) != EXPECTED_SNPS:
        raise ValueError(
            f"{args.vcf}: expected {EXPECTED_SNPS} mapped SNP records, found {len(aims_rsids)}"
        )

    genotypes_output = args.genotypes_output or DATA_DIR / f"aims56_CHN100K_TOP{args.chn100k_top}_1000g_2504_genotypes.csv"
    alleles_output = args.alleles_output or DATA_DIR / f"aims56_CHN100K_TOP{args.chn100k_top}_1000g_alleles.csv"

    genotype_fields = ["SampleID", "Superpopulation", "Population", *rsids]

    def genotype_rows() -> Iterable[dict[str, str]]:
        for sample_id, genotypes in zip(projected_samples, genotypes_by_sample):
            superpopulation, population = population_by_sample[sample_id]
            yield {
                "SampleID": sample_id,
                "Superpopulation": superpopulation,
                "Population": population,
                **dict(zip(rsids, genotypes)),
            }

    write_csv_atomic(genotypes_output, genotype_fields, genotype_rows())
    write_csv_atomic(
        alleles_output,
        ["rsID", "CHROM", "POS", "REF", "ALT"],
        alleles,
    )

    print(f"Population table samples: {len(population_by_sample)}")
    print(f"Projected unrelated samples: {len(projected_samples)}")
    print(f"Mapped VCF SNPs: {len(rsids)}")
    print(f"CHN100K top per chromosome: {args.chn100k_top}")
    print(f"Genotype CSV: {genotypes_output}")
    print(f"Allele CSV (REF/ALT from VCF): {alleles_output}")


if __name__ == "__main__":
    main()
