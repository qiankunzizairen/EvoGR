import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .fasta import normalize_species_name, read_fasta
from .windows import iter_windows

CLASS_IDS = {"human": 0, "non_human_animal": 1, "plant": 2, "microorganism": 3}
NON_HUMAN_FOLDS = {
    "non_human_animal": [
        ["Pan troglodytes", "Ciona intestinalis"], ["Apis mellifera", "Gallus gallus"],
        ["Mus musculus", "Strongylocentrotus purpuratus"], ["Xenopus tropicalis", "Danio rerio"],
        ["Caenorhabditis elegans", "Drosophila melanogaster"],
    ],
    "plant": [
        ["Vitis vinifera", "Solanum tuberosum"], ["Glycine max", "Zea mays"],
        ["Oryza sativa", "Populus trichocarpa"], ["Sorghum bicolor", "Solanum lycopersicum"],
        ["Arabidopsis thaliana", "Hordeum vulgare"],
    ],
    "microorganism": [
        ["Methanobrevibacter smithii ATCC 35061", "Methanosarcina mazei S-6", "Thermus thermophilus HB8", "Vibrio cholerae", "Thermotoga maritima MSB8", "Escherichia coli str. K-12 substr. MG1655", "Mycobacterium tuberculosis H37Rv", "Fusobacterium nucleatum subsp. nucleatum ATCC 25586"],
        ["Thermococcus kodakarensis KOD1", "Sulfolobus acidocaldarius DSM 639", "Caulobacter vibrioides NA1000", "Listeria monocytogenes EGD-e", "Pseudomonas aeruginosa PAO1", "Verrucomicrobium spinosum DSM 4136 = JCM 18804", "Corynebacterium glutamicum ATCC 13032", "Salmonella enterica subsp. enterica serovar Typhimurium str. LT2"],
        ["Haloferax volcanii DS2", "Archaeoglobus fulgidus DSM 4304", "Streptomyces coelicolor A3(2)", "Geobacter sulfurreducens PCA", "Planctopirus limnophila DSM 3776", "Synechococcus elongatus PCC 7942 = FACHB-805", "Chlorobaculum tepidum TLS", "Coxiella burnetii RSA 493"],
        ["Methanococcus maripaludis", "Pyrococcus furiosus DSM 3638", "Acidobacterium capsulatum ATCC 51196", "Streptococcus pneumoniae", "Deinococcus radiodurans R1 = ATCC 13939 = DSM 20539", "Bacteroides thetaiotaomicron VPI-5482", "Staphylococcus aureus subsp. aureus NCTC 8325", "Bacillus subtilis subsp. subtilis str. 168"],
        ["Methanocaldococcus jannaschii DSM 2661", "Nitrosopumilus maritimus SCM1", "Aquifex aeolicus VF5", "Borreliella burgdorferi", "Treponema pallidum subsp. pallidum str. Nichols", "Chlamydia trachomatis D/UW-3/CX", "Campylobacter jejuni subsp. jejuni NCTC 11168 = ATCC 700819", "Helicobacter pylori"],
    ],
}
HUMAN_FOLDS = [["chr2", "chr16", "chr17", "chr21", "chrX"], ["chr4", "chr5", "chr11", "chr19", "chr22"],
               ["chr6", "chr8", "chr13", "chr14", "chr18"], ["chr1", "chr10", "chr12", "chr15"],
               ["chr3", "chr7", "chr9", "chr20", "chrY"]]
FIELDS = ["window_id", "class_name", "class_id", "species", "source_fasta", "sequence_id", "sequence_name", "start", "end", "length_bp"]

BASE_FOLD_BY_NAME = {name: fold for folds in NON_HUMAN_FOLDS.values() for fold, names in enumerate(folds) for name in names}
BASE_FOLD_BY_NORMALIZED = {normalize_species_name(name): fold for name, fold in BASE_FOLD_BY_NAME.items()}
HUMAN_FOLD_BY_CHROMOSOME = {chrom: fold for fold, chroms in enumerate(HUMAN_FOLDS) for chrom in chroms}


def assign_base_fold(row: dict[str, str]) -> int:
    if row["class_name"] == "human":
        sequence_name = row["sequence_name"].strip()
        chromosome = sequence_name if sequence_name in HUMAN_FOLD_BY_CHROMOSOME else sequence_name.split()[0]
        try:
            return HUMAN_FOLD_BY_CHROMOSOME[chromosome]
        except KeyError as exc:
            raise ValueError(f"Unknown human chromosome: {sequence_name!r}") from exc
    species = row["species"].strip()
    fold = BASE_FOLD_BY_NAME.get(species)
    if fold is None:
        fold = BASE_FOLD_BY_NORMALIZED.get(normalize_species_name(species))
    if fold is None:
        raise ValueError(f"Unknown species for fixed folds: {species!r}")
    return fold


def build_splits_from_manifest(manifest_path: Path, output_root: Path) -> dict[tuple[int, str], int]:
    output_root.mkdir(parents=True, exist_ok=True)
    handles, writers, counts = {}, {}, {}
    fields = FIELDS + ["base_fold", "experiment_fold", "split"]
    try:
        for fold in range(5):
            for split in ("train", "val", "test"):
                key = (fold, split)
                handle = (output_root / f"fold_{fold}_{split}.csv").open("w", newline="")
                handles[key] = handle
                writers[key] = csv.DictWriter(handle, fieldnames=fields)
                writers[key].writeheader(); counts[key] = 0
        seen = set(); total = 0
        with manifest_path.open(newline="") as source:
            reader = csv.DictReader(source)
            missing = [field for field in FIELDS if field not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"Manifest missing required fields: {', '.join(missing)}")
            for row in reader:
                window_id = row["window_id"]
                if window_id in seen:
                    raise ValueError(f"Duplicate window_id: {window_id}")
                seen.add(window_id); total += 1
                base = assign_base_fold(row)
                for experiment in range(5):
                    split = "test" if base == experiment else "val" if base == (experiment + 1) % 5 else "train"
                    out = {field: row[field] for field in FIELDS}
                    out.update(base_fold=str(base), experiment_fold=str(experiment), split=split)
                    writers[(experiment, split)].writerow(out); counts[(experiment, split)] += 1
        if total == 0:
            raise ValueError("Manifest contains no windows")
        return counts
    finally:
        for handle in handles.values():
            handle.close()


def load_species(class_name: str, input_root: Path) -> dict[str, Path]:
    if class_name == "human":
        files = list((input_root / "human").glob("*.fna"))
        if len(files) != 1:
            raise ValueError(f"Expected one human FASTA, found {len(files)}")
        return {"human": files[0]}
    csv_name = "animal_gcf_accessions.csv" if class_name == "non_human_animal" else f"{class_name}_gcf_accessions.csv"
    csv_path = input_root / csv_name
    rows = list(csv.DictReader(csv_path.open(newline="")))
    files = {p.stem: p for p in (input_root / class_name).glob("*.fna")}
    result = {}
    for row in rows:
        species = row["Species"].strip()
        key = normalize_species_name(species)
        if key not in files:
            raise ValueError(f"No FASTA for species {species!r} (expected {key}.fna)")
        result[species] = files[key]
    expected = 40 if class_name == "microorganism" else 10
    if len(result) != expected or len(files) != expected:
        raise ValueError(f"Expected {expected} {class_name} species and FASTA files")
    return result


def build_manifests(input_root: Path, output_root: Path, window_size: int, stride: int, max_n_fraction: float) -> dict[tuple[int, str], int]:
    groups: dict[tuple[str, str], int] = {}
    records = []
    genome_count = 0
    for class_name in CLASS_IDS:
        files = load_species(class_name, input_root)
        genome_count += len(files)
        if class_name == "human":
            fold_map = {chrom: fold for fold, chroms in enumerate(HUMAN_FOLDS) for chrom in chroms}
            species_items = [("human", files["human"])]
        else:
            fold_map = {species: fold for fold, species_list in enumerate(NON_HUMAN_FOLDS[class_name]) for species in species_list}
            species_items = list(files.items())
        for species, fasta_path in species_items:
            try:
                source = fasta_path.relative_to(Path.cwd()).as_posix()
            except ValueError:
                source = fasta_path.as_posix()
            for record in read_fasta(fasta_path):
                if class_name == "human":
                    chrom = record.sequence_id if record.sequence_id in fold_map else re.match(r"(chr(?:[1-9]|1[0-9]|2[0-2]|X|Y))(?:$|\s)", record.sequence_name)
                    chrom = chrom.group(1) if hasattr(chrom, "group") else chrom
                    if chrom not in fold_map:
                        continue
                    group = chrom
                    base_fold = fold_map[chrom]
                else:
                    group, base_fold = species, fold_map[species]
                groups[(class_name, group)] = base_fold
                for window in iter_windows(record, window_size, stride, max_n_fraction):
                    window_id = f"{class_name}:{species}:{window.sequence_id}:{window.start}:{window.end}"
                    row = {"window_id": window_id, "class_name": class_name, "class_id": CLASS_IDS[class_name], "species": species,
                           "source_fasta": source, "sequence_id": window.sequence_id, "sequence_name": window.sequence_name,
                           "start": window.start, "end": window.end, "length_bp": window.end - window.start, "base_fold": base_fold}
                    records.append(row)
    if genome_count != 61:
        raise ValueError(f"Expected 61 genomes, found {genome_count}")
    output_root.mkdir(parents=True, exist_ok=True)
    counts = {}
    for fold in range(5):
        for split in ("train", "val", "test"):
            rows = [r for r in records if (r["base_fold"] == fold if split == "test" else r["base_fold"] == (fold + 1) % 5 if split == "val" else r["base_fold"] not in (fold, (fold + 1) % 5))]
            path = output_root / f"fold_{fold}_{split}.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows({k: row[k] for k in FIELDS} for row in rows)
            counts[(fold, split)] = len(rows)
    return counts
