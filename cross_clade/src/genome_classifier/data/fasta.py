from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import re


@dataclass
class FastaRecord:
    sequence_id: str
    sequence_name: str
    sequence: str


def normalize_species_name(species: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", species).strip("_")


def read_fasta(path: Path) -> Iterator[FastaRecord]:
    name = None
    chunks = []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield _record(name, chunks)
                name = line[1:].strip()
                chunks = []
            else:
                if name is None:
                    raise ValueError(f"Sequence data precedes FASTA header in {path}")
                chunks.append(line)
        if name is not None:
            yield _record(name, chunks)


def _record(header: str, chunks: list[str]) -> FastaRecord:
    sequence_id = header.split()[0]
    return FastaRecord(sequence_id, header, "".join(chunks).upper())

