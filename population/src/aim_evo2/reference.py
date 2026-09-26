from __future__ import annotations

from pathlib import Path


def normalize_chrom(value: str) -> str:
    value = value.strip()
    if value.lower().startswith("chr"):
        value = value[3:]
    return value


class IndexedFasta:
    def __init__(self, fasta_path: Path, fai_path: Path | None = None):
        self.fasta_path = Path(fasta_path)
        self.fai_path = Path(fai_path) if fai_path is not None else self.fasta_path.with_suffix(self.fasta_path.suffix + ".fai")
        if not self.fasta_path.exists():
            raise FileNotFoundError(self.fasta_path)
        if not self.fai_path.exists():
            raise FileNotFoundError(self.fai_path)
        self.records = self._read_fai()
        self.aliases = {normalize_chrom(name): name for name in self.records}

    def _read_fai(self) -> dict[str, tuple[int, int, int, int]]:
        records: dict[str, tuple[int, int, int, int]] = {}
        with self.fai_path.open(encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 5:
                    continue
                name, length, offset, line_bases, line_width = fields[:5]
                records[name] = (int(length), int(offset), int(line_bases), int(line_width))
        return records

    def resolve_chrom(self, chrom: str) -> str:
        if chrom in self.records:
            return chrom
        normalized = normalize_chrom(chrom)
        if normalized in self.aliases:
            return self.aliases[normalized]
        prefixed = f"chr{normalized}"
        if prefixed in self.records:
            return prefixed
        raise KeyError(f"Chromosome {chrom!r} not found in FASTA index")

    def fetch(self, chrom: str, start0: int, end0: int) -> str:
        name = self.resolve_chrom(chrom)
        length, offset, line_bases, line_width = self.records[name]
        if start0 < 0 or end0 > length or start0 >= end0:
            raise ValueError(f"Invalid interval {chrom}:{start0}-{end0} for length {length}")
        chunks: list[bytes] = []
        remaining = end0 - start0
        pos = start0
        with self.fasta_path.open("rb") as handle:
            while remaining:
                line_index = pos // line_bases
                within = pos % line_bases
                readable = min(remaining, line_bases - within)
                byte_offset = offset + line_index * line_width + within
                handle.seek(byte_offset)
                chunks.append(handle.read(readable))
                pos += readable
                remaining -= readable
        return b"".join(chunks).decode("ascii").upper()
