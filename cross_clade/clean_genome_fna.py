#!/usr/bin/env python3
from pathlib import Path
import argparse
import os
import re
import sys
import tempfile


PRIMARY_HUMAN = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
PLASMID_RE = re.compile(r"\b(?:plasmid|megaplasmid)\b")


def classify_header(header, group):
    text = header.lower()
    fields = header.split()
    if not fields or not fields[0].startswith(">") or fields[0] == ">":
        return False, "invalid_header"
    seq_id = fields[0][1:]

    if group == "human":
        if seq_id == "chrM" or "rl:mitochondrion" in text:
            return False, "mitochondrion"
        if seq_id in PRIMARY_HUMAN and "as:grch38" in text:
            return True, "primary_chromosome"
        if (
            ("rl:unlocalized" in text or "rl:unplaced" in text)
            and "as:grch38" in text
        ):
            return True, "primary_unlocalized_unplaced"
        return False, "alt_patch_decoy_hla_or_other"

    if group == "plant":
        if "chloroplast" in text or "plastid" in text:
            return False, "chloroplast_or_plastid"
        if "mitochond" in text:
            return False, "mitochondrion"
        if " chromosome " in f" {text} ":
            return True, "nuclear_chromosome"
        if "unlocalized genomic scaffold" in text or "unplaced genomic scaffold" in text:
            return True, "nuclear_unlocalized_unplaced"
        return False, "non_nuclear_or_unknown"

    if group == "non_human_animal":
        if "mitochond" in text:
            return False, "mitochondrion"
        if " chromosome " in f" {text} " or "linkage group" in text:
            return True, "nuclear_chromosome_or_linkage_group"
        if "unlocalized genomic scaffold" in text or "unplaced genomic scaffold" in text:
            return True, "nuclear_unlocalized_unplaced"
        # The RefSeq D. melanogaster file in this data set contains nuclear NW_
        # scaffolds whose definition lines end only in "sequence".
        if seq_id.startswith("NW_") and "drosophila melanogaster" in text:
            return True, "nuclear_drosophila_scaffold"
        return False, "non_nuclear_or_unknown"

    if group == "microorganism":
        if PLASMID_RE.search(text):
            return False, "plasmid"
        if (
            " chromosome " in f" {text} "
            or "complete genome" in text
            or "complete sequence" in text
        ):
            return True, "chromosomal_replicon"
        return False, "non_chromosomal_or_unknown_replicon"

    raise ValueError(f"unsupported group: {group}")


def scan_fasta(path, group):
    kept = removed = total = 0
    reasons = {}
    with path.open("r", errors="replace") as in_handle:
        for line in in_handle:
            if not line.startswith(">"):
                continue
            total += 1
            keep, reason = classify_header(line.strip(), group)
            if keep:
                kept += 1
            else:
                removed += 1
                reasons[reason] = reasons.get(reason, 0) + 1

    return {
        "path": path,
        "total": total,
        "kept": kept,
        "removed": removed,
        "reasons": reasons,
        "updated": False,
    }


def filter_fasta(path, group, dry_run=False):
    summary = scan_fasta(path, group)

    # Do not rewrite an already-clean file, and never replace an input with an
    # empty result. main() reports the latter as an error.
    if dry_run or summary["removed"] == 0 or summary["kept"] == 0:
        return summary

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    out_handle = os.fdopen(fd, "w")
    keep_current = False

    try:
        with path.open("r", errors="replace") as in_handle:
            for line in in_handle:
                if line.startswith(">"):
                    keep_current, _ = classify_header(line.strip(), group)
                if keep_current:
                    out_handle.write(line)

        out_handle.close()
        os.replace(tmp_name, path)
        tmp_name = None
        summary["updated"] = True
    finally:
        if not out_handle.closed:
            out_handle.close()
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass

    return summary


def find_assembly_dir(script_dir, explicit):
    if explicit:
        return Path(explicit).expanduser().resolve()
    return script_dir / "data" / "assembly"


def main():
    parser = argparse.ArgumentParser(
        description="Clean genome FASTA files in place by removing non-target replicons."
    )
    parser.add_argument(
        "--assembly-dir",
        help="Assembly directory containing human, plant, non_human_animal, microorganism.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be removed without modifying files.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    assembly_dir = find_assembly_dir(script_dir, args.assembly_dir)
    groups = ["human", "plant", "non_human_animal", "microorganism"]

    if not assembly_dir.is_dir():
        print(f"ERROR: assembly directory not found: {assembly_dir}", file=sys.stderr)
        return 1

    roots = {group: assembly_dir / group for group in groups}
    missing = [root for root in roots.values() if not root.is_dir()]
    if missing:
        for root in missing:
            print(f"ERROR: missing group directory: {root}", file=sys.stderr)
        return 1

    summaries = []
    for group, root in roots.items():
        for path in sorted(root.glob("*.fna")):
            summaries.append(filter_fasta(path, group, dry_run=args.dry_run))

    mode = "DRY_RUN" if args.dry_run else "UPDATED"
    print(f"{mode}\tassembly_dir={assembly_dir}")
    for item in summaries:
        reason_text = ",".join(
            f"{reason}:{count}" for reason, count in sorted(item["reasons"].items())
        )
        if not reason_text:
            reason_text = "none"
        rel = item["path"].relative_to(assembly_dir)
        print(
            f"{rel}\ttotal={item['total']}\tkept={item['kept']}"
            f"\tremoved={item['removed']}\treasons={reason_text}"
        )

    empty = [item for item in summaries if item["kept"] == 0]
    if empty:
        print(
            "ERROR: at least one FASTA would be empty after filtering; "
            "the affected file was not modified:",
            file=sys.stderr,
        )
        for item in empty:
            print(f"  {item['path']}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
