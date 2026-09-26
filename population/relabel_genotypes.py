#!/usr/bin/env python3
"""Create a Label-labelled genotype CSV from the source matrix.

The TOP value is intentionally not hard-coded: by default the script finds
the matching ``aims56_CHN100K_TOP{x}_1000g_2504_genotypes.csv`` in ``data``.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


SOURCE_RE = re.compile(r"^aims56_CHN100K_TOP(?P<x>[^_]+)_1000g_2504_genotypes\.csv$")


def read_mapping(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        if not rows.fieldnames or not {"Label", "Group"}.issubset(rows.fieldnames):
            raise ValueError(f"metadata must contain Label and Group columns: {path}")
        mapping: dict[str, str] = {}
        for row in rows:
            label = row["Label"].strip()
            for group in row["Group"].split(","):
                group = group.strip()
                if group:
                    if group in mapping and mapping[group] != label:
                        raise ValueError(f"conflicting mapping for group {group!r}")
                    mapping[group] = label
        return mapping


def find_source(data_dir: Path, source: Path | None) -> Path:
    if source:
        return source
    candidates = sorted(p for p in data_dir.iterdir() if SOURCE_RE.match(p.name))
    if len(candidates) != 1:
        names = ", ".join(p.name for p in candidates) or "none"
        raise ValueError(f"expected exactly one TOPx genotype CSV, found: {names}")
    return candidates[0]


def convert(source: Path, output: Path, metadata: Path) -> tuple[int, int]:
    mapping = read_mapping(metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    unmapped: list[tuple[int, str, str, str]] = []
    count = 0
    with source.open(newline="", encoding="utf-8-sig") as src, output.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        required = {"SampleID", "Superpopulation", "Population"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"source must contain {sorted(required)} columns: {source}")
        genotype_columns = [c for c in reader.fieldnames if c not in required]
        writer = csv.DictWriter(dst, fieldnames=["SampleID", "Label", *genotype_columns])
        writer.writeheader()
        for line_no, row in enumerate(reader, start=2):
            superpop = row["Superpopulation"].strip()
            population = row["Population"].strip()
            label = mapping.get(superpop)
            if label is None:
                # Population-specific groups (for example CDX/CHB/CHS) take
                # effect only when Superpopulation itself is not mapped.
                label = mapping.get(population)
            if label is None:
                unmapped.append((line_no, row["SampleID"], superpop, population))
                continue
            writer.writerow({"SampleID": row["SampleID"], "Label": label, **{c: row[c] for c in genotype_columns}})
            count += 1
    return count, len(unmapped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--metadata", type=Path, default=Path("data/aims56_1000g_5_Label_metadata.tsv"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = find_source(args.data_dir, args.source)
    default_name = re.sub(r"_2504_genotypes\.csv$", "_5_Label.csv", source.name)
    output = args.output or source.with_name(default_name)
    count, ignored = convert(source, output, args.metadata)
    print(f"Wrote {count} mapped rows to {output}; ignored {ignored} unmapped rows")


if __name__ == "__main__":
    main()
