"""Input handling for read based population classification.

Each text file is one sample.  Files may use a small FASTA-like format
(`>read_id` followed by one or more sequence lines) or simply contain one
sequence per non-empty line.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


DNA_BASES = frozenset("ACGTN")


@dataclass(frozen=True)
class ReadSample:
    path: Path
    reads: tuple[str, ...]
    label: int


def read_txt_reads(path: Path) -> tuple[str, ...]:
    """Read variable-length single-stranded reads from *path*.

    FASTA records are supported, including wrapped sequence lines.  When no
    header is present, every non-empty line is treated as one read.
    """
    records: list[str] = []
    current: list[str] | None = None
    saw_header = False
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                saw_header = True
                if current is not None:
                    records.append("".join(current))
                current = []
                continue
            sequence = "".join(line.split()).upper()
            if not sequence or any(base not in DNA_BASES for base in sequence):
                raise ValueError(f"{path}:{line_number}: invalid DNA sequence")
            if saw_header:
                if current is None:
                    raise ValueError(f"{path}:{line_number}: sequence appears before a FASTA header")
                current.append(sequence)
            else:
                records.append(sequence)
    if current is not None:
        records.append("".join(current))
    if not records or any(not read for read in records):
        raise ValueError(f"{path}: at least one non-empty read is required")
    return tuple(records)


def read_sample_mapping(path: Path) -> dict[str, int | str]:
    """Read a two-column mapping file: txt filename, label.

    A header is optional.  Comma, tab, and whitespace-separated files are
    accepted.  Labels may be strings or integers and are encoded later.
    """
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw_lines = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]
    if not raw_lines:
        raise ValueError(f"{path}: mapping file is empty")
    delimiter = "\t" if "\t" in raw_lines[0] else "," if "," in raw_lines[0] else None
    for line_number, line in enumerate(raw_lines, start=1):
        fields = next(csv.reader([line], delimiter=delimiter or " ")) if delimiter else line.split()
        fields = [field.strip() for field in fields if field.strip()]
        if line_number == 1 and len(fields) >= 2 and fields[0].lower() in {"file", "filename", "sample", "sample_id"}:
            continue
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected FILE and LABEL columns")
        filename, label = fields
        if filename in {name for name, _ in rows}:
            raise ValueError(f"{path}:{line_number}: duplicate filename {filename!r}")
        rows.append((filename, label))
    if not rows:
        raise ValueError(f"{path}: mapping file contains no samples")
    return {filename: label for filename, label in rows}


def load_read_samples(read_dir: Path, mapping_path: Path, num_classes: int | None = None) -> tuple[list[ReadSample], dict[str, int]]:
    mapping = read_sample_mapping(mapping_path)
    files = {path.name: path for path in read_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt"}
    files.update({str(path.relative_to(read_dir)): path for path in read_dir.rglob("*.txt") if path.is_file()})
    missing = sorted(set(mapping) - set(files))
    if missing:
        raise FileNotFoundError(f"mapping references missing txt files: {', '.join(missing)}")
    labels = sorted(set(str(value) for value in mapping.values()))
    if len(labels) < 2:
        raise ValueError("mapping must contain at least two population classes")
    if num_classes is not None and len(labels) > num_classes:
        raise ValueError(f"mapping contains {len(labels)} classes, exceeding --num-classes={num_classes}")
    label_to_id = {label: index for index, label in enumerate(labels)}
    samples = [ReadSample(files[name], read_txt_reads(files[name]), label_to_id[str(mapping[name])]) for name in sorted(mapping)]
    if num_classes is not None and num_classes < 2:
        raise ValueError("--num-classes must be at least 2")
    return samples, label_to_id
