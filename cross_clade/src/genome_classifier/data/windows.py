import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .fasta import FastaRecord, normalize_species_name, read_fasta

CLASS_IDS = {"human": 0, "non_human_animal": 1, "plant": 2, "microorganism": 3}
WINDOW_FIELDS = ["window_id", "class_name", "class_id", "species", "source_fasta", "sequence_id", "sequence_name", "start", "end", "length_bp"]


@dataclass(frozen=True)
class Window:
    sequence_id: str
    sequence_name: str
    start: int
    end: int


def is_canonical_human_chromosome(sequence_id: str, sequence_name: str | None = None) -> bool:
    """Return whether a FASTA record identifies one GRCh38 canonical chromosome."""
    candidate = sequence_id
    if candidate not in canonical_human_chromosomes() and sequence_name:
        candidate = sequence_name.split()[0]
    return candidate in canonical_human_chromosomes()


def canonical_human_chromosomes() -> tuple[str, ...]:
    return tuple([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])


def iter_windows(record: FastaRecord, window_size: int = 8192, stride: int = 8192,
                 max_n_fraction: float = 0.05) -> Iterator[Window]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    for start in range(0, max(0, len(record.sequence) - window_size + 1), stride):
        end = start + window_size
        sequence = record.sequence[start:end]
        if sequence.count("N") / window_size > max_n_fraction:
            continue
        yield Window(record.sequence_id, record.sequence_name, start, end)


def load_species(class_name: str, input_root: Path) -> list[tuple[str, Path]]:
    if class_name == "human":
        files = sorted((input_root / "human").glob("*.fna"))
        if len(files) != 1:
            raise ValueError(f"Expected one human FASTA, found {len(files)}")
        return [("Homo sapiens", files[0])]
    csv_name = "animal_gcf_accessions.csv" if class_name == "non_human_animal" else f"{class_name}_gcf_accessions.csv"
    with (input_root / csv_name).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    files = {path.stem: path for path in (input_root / class_name).glob("*.fna")}
    result = []
    for row in rows:
        species = row["Species"].strip()
        key = normalize_species_name(species)
        if key not in files:
            raise ValueError(f"No FASTA for species {species!r} (expected {key}.fna)")
        result.append((species, files[key]))
    if len(result) != len(files):
        raise ValueError(f"CSV and FASTA counts differ for {class_name}")
    return sorted(result, key=lambda item: item[0])


def build_manifest(input_root: Path, output_file: Path, window_size: int = 8192,
                   stride: int = 8192, max_n_fraction: float = 0.05) -> dict[str, int]:
    rows = []
    counts = {name: 0 for name in CLASS_IDS}
    canonical = set(canonical_human_chromosomes())
    genome_count = 0
    project_root = input_root.resolve().parent.parent
    for class_name in CLASS_IDS:
        for species, fasta_path in load_species(class_name, input_root):
            genome_count += 1
            try:
                source = fasta_path.resolve().relative_to(project_root).as_posix()
            except ValueError:
                source = fasta_path.as_posix()
            for record in read_fasta(fasta_path):
                if class_name == "human" and record.sequence_id not in canonical:
                    continue
                sequence_name = record.sequence_id if class_name == "human" else record.sequence_name
                for window in iter_windows(record, window_size, stride, max_n_fraction):
                    rows.append({"window_id": f"{class_name}:{species}:{window.sequence_id}:{window.start}:{window.end}",
                                 "class_name": class_name, "class_id": CLASS_IDS[class_name], "species": species,
                                 "source_fasta": source, "sequence_id": window.sequence_id,
                                 "sequence_name": sequence_name, "start": window.start, "end": window.end,
                                 "length_bp": window.end - window.start})
                    counts[class_name] += 1
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WINDOW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if len({row["window_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate window_id detected")
    return counts
