from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import resolve_path

VALID_GT = {"0|0", "0|1", "1|0", "1|1"}
DNA_BASES = {"A", "C", "G", "T"}
RSID_RE = re.compile(r"^rs\d+$")


@dataclass(frozen=True)
class ValidatedData:
    genotypes: pd.DataFrame
    alleles: pd.DataFrame
    label_mapping: dict[int, str]
    rsid_order: list[str]
    n_aim: int


def expected_n_aim(config: dict) -> int:
    data = config["data"]
    return int(data["base_aim_count"]) + int(data["autosome_count"]) * int(data["added_aims_per_autosome"])


def read_label_metadata(path: Path) -> dict[int, str]:
    frame = pd.read_csv(path, sep="\t", dtype=str)
    if list(frame.columns) != ["Label", "Group"]:
        raise ValueError(f"{path}: expected Label,Group header, found {list(frame.columns)}")
    mapping = {int(row.Label): str(row.Group).strip() for row in frame.itertuples(index=False)}
    if len(mapping) != len(frame) or set(mapping) != set(range(5)) or not all(mapping.values()):
        raise ValueError(f"{path}: metadata must uniquely map labels 0..4 to non-empty groups")
    return mapping


def read_alleles(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str)
    expected = ["rsID", "CHROM", "POS", "REF", "ALT"]
    if list(frame.columns) != expected:
        raise ValueError(f"{path}: expected header {expected}, found {list(frame.columns)}")
    if not frame["rsID"].is_unique:
        raise ValueError(f"{path}: duplicate rsID {frame.loc[frame.rsID.duplicated(), 'rsID'].iloc[0]}")
    for i, row in frame.iterrows():
        rsid, ref, alt = str(row.rsID), str(row.REF).upper(), str(row.ALT).upper()
        try: valid_pos = int(str(row.POS)) > 0
        except ValueError: valid_pos = False
        if not RSID_RE.fullmatch(rsid) or not valid_pos or ref not in DNA_BASES or alt not in DNA_BASES or ref == alt:
            raise ValueError(f"{path}:{i + 2}: invalid biallelic SNV record {rsid} {row.POS} {ref}/{alt}")
    return frame.assign(REF=frame.REF.str.upper(), ALT=frame.ALT.str.upper())


def read_genotypes(path: Path, mapping: dict[int, str]) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path, dtype=str)
    if list(frame.columns[:2]) != ["SampleID", "Label"]:
        raise ValueError(f"{path}: first columns must be SampleID,Label; found {list(frame.columns[:2])}")
    if not frame.SampleID.is_unique:
        raise ValueError(f"{path}: duplicate SampleID {frame.loc[frame.SampleID.duplicated(), 'SampleID'].iloc[0]}")
    rsids = list(frame.columns[2:])
    if not rsids or len(rsids) != len(set(rsids)) or any(not RSID_RE.fullmatch(x) for x in rsids):
        raise ValueError(f"{path}: genotype columns must be unique rsIDs")
    try: frame["Label"] = frame.Label.astype(int)
    except ValueError as exc: raise ValueError(f"{path}: labels must be integers") from exc
    if set(frame.Label) != set(mapping): raise ValueError(f"{path}: observed labels must cover 0..4")
    for rsid in rsids:
        bad = ~frame[rsid].isin(VALID_GT)
        if bad.any():
            row = frame.loc[bad].iloc[0]
            raise ValueError(f"{path}: invalid genotype SampleID={row.SampleID} rsID={rsid} raw_value={row[rsid]!r}")
    return frame, rsids


def load_validated_data(config: dict) -> ValidatedData:
    paths = {
        key: resolve_path(config, config["data"][key])
        for key in ("genotype_csv", "label_metadata_tsv", "allele_metadata_csv", "reference_fasta", "reference_fai")
    }
    missing = [f"{key}={path}" for key, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required data files are missing: " + ", ".join(missing))
    mapping = read_label_metadata(paths["label_metadata_tsv"])
    alleles = read_alleles(paths["allele_metadata_csv"])
    genotypes, rsids = read_genotypes(paths["genotype_csv"], mapping)
    expected = expected_n_aim(config)
    actual_n_aim = len(rsids)
    if actual_n_aim != expected or len(alleles) != expected or actual_n_aim != len(alleles):
        raise ValueError(f"N_AIM mismatch: expected={expected}, genotype_loci={len(rsids)}, allele_rows={len(alleles)}")
    if set(rsids) != set(alleles.rsID): raise ValueError("Genotype rsID columns and allele metadata rsIDs differ")
    return ValidatedData(genotypes, alleles.set_index("rsID").loc[rsids].reset_index(), mapping, rsids, actual_n_aim)


def genotype_to_dosage(value: str) -> int:
    if value not in VALID_GT: raise ValueError(f"Invalid genotype {value!r}")
    return int(value[0]) + int(value[2])
