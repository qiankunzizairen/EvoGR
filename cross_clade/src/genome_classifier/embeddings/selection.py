from __future__ import annotations
import csv, random, hashlib
from pathlib import Path
from collections import defaultdict

FIELDS = ["window_id", "class_name", "class_id", "species", "source_fasta", "sequence_id", "sequence_name", "start", "end", "length_bp"]
CLASSES = ("human", "non_human_animal", "plant", "microorganism")

def _group(row):
    return row["sequence_id"] if row["class_name"] == "human" else row["species"]

def _rng(seed, key):
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))

def select_rows(rows, quota, seed=42):
    """Deterministically select quota rows with group round-robin and no replacement."""
    groups = defaultdict(list)
    for row in rows:
        groups[_group(row)].append(dict(row))
    if not groups:
        raise ValueError("No rows available for selection")
    for key, values in groups.items():
        _rng(seed, key).shuffle(values)
    selected = []
    keys = sorted(groups)
    positions = {k: 0 for k in keys}
    while len(selected) < quota:
        progressed = False
        for key in keys:
            pos = positions[key]
            if pos < len(groups[key]):
                selected.append(groups[key][pos]); positions[key] = pos + 1; progressed = True
                if len(selected) == quota: break
        if not progressed:
            raise ValueError(f"Insufficient windows for quota {quota}; found {len(selected)}")
    return selected

def build_selection(split_dir: Path, output: Path, quotas: dict, seed=42):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment_fold", "split"] + FIELDS); writer.writeheader()
        seen = set()
        for fold in range(5):
            for split in ("train", "val", "test"):
                quota = quotas[f"{split}_per_class"]
                by_class = defaultdict(lambda: defaultdict(list))
                path = split_dir / f"fold_{fold}_{split}.csv"
                with path.open(newline="") as source:
                    reader = csv.DictReader(source)
                    missing = [f for f in FIELDS if f not in (reader.fieldnames or [])]
                    if missing: raise ValueError(f"{path} missing fields: {', '.join(missing)}")
                    seen_counts = defaultdict(lambda: defaultdict(int))
                    for row in reader:
                        c = row["class_name"]; g = _group(row); seen_counts[c][g] += 1
                        bucket = by_class[c][g]
                        if len(bucket) < quota: bucket.append(row)
                        else:
                            j = _rng(seed + fold * 100 + (0 if split == "train" else 1 if split == "val" else 2), f"reservoir:{c}:{g}:{seen_counts[c][g]}").randrange(seen_counts[c][g])
                            if j < quota: bucket[j] = row
                for class_name in CLASSES:
                    candidates = [r for values in by_class[class_name].values() for r in values]
                    picked = select_rows(candidates, quota, seed + fold * 100 + (0 if split == "train" else 1 if split == "val" else 2))
                    for row in picked:
                        key = (fold, split, row["window_id"])
                        if key in seen: raise ValueError(f"Duplicate selection: {key}")
                        seen.add(key); writer.writerow({"experiment_fold": fold, "split": split, **{f: row[f] for f in FIELDS}})
