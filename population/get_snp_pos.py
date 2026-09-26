from pathlib import Path
import requests
import pandas as pd

INPUT_FILE = Path("data/aims56.txt")
OUTPUT_FILE = Path("data/aims56_grch38.csv")

ENSEMBL_REST = "https://rest.ensembl.org"
SPECIES = "homo_sapiens"

TARGET_ASSEMBLY = "GRCh38"

CANONICAL_CHROMS = {str(i) for i in range(1, 23)} | {"X", "Y", "MT"}

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    rsids = [
        line.strip()
        for line in f
        if line.strip()
    ]

rsids = list(dict.fromkeys(rsids))

print(f"Loaded {len(rsids)} unique rsIDs.")

if len(rsids) != 56:
    print(f"WARNING: Expected 56 SNPs, but loaded {len(rsids)}.")

invalid_rsids = [x for x in rsids if not x.lower().startswith("rs")]
if invalid_rsids:
    raise ValueError(f"Invalid rsID format: {invalid_rsids}")

url = f"{ENSEMBL_REST}/variation/{SPECIES}"

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}

payload = {
    "ids": rsids
}

response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=60
)

response.raise_for_status()

data = response.json()

print(f"Ensembl returned {len(data)} variant records.")

rows = []

for rsid in rsids:

    record = data.get(rsid)

    if record is None:
        print(f"WARNING: {rsid} was not found in Ensembl.")
        rows.append({
            "rsID": rsid,
            "CHROM": None,
            "POS": None,
            "REF": None,
            "ALT": None,
        })
        continue

    mappings = record.get("mappings", [])

    grch38_mappings = [
        m for m in mappings
        if m.get("assembly_name") == TARGET_ASSEMBLY
        and str(m.get("seq_region_name")) in CANONICAL_CHROMS
        and m.get("start") == m.get("end")
    ]

    if len(grch38_mappings) == 0:
        print(f"WARNING: No GRCh38 SNP mapping on a canonical chromosome was found for {rsid}.")
        rows.append({
            "rsID": rsid,
            "CHROM": None,
            "POS": None,
            "REF": None,
            "ALT": None,
        })
        continue

    if len(grch38_mappings) > 1:
        print(
            f"WARNING: Multiple canonical GRCh38 mappings were found for {rsid}; "
            f"using the first one."
        )

    mapping = grch38_mappings[0]

    chrom = str(mapping["seq_region_name"])
    pos = int(mapping["start"])

    allele_string = mapping.get("allele_string")

    ref = None
    alt = None

    if allele_string:
        alleles = allele_string.split("/")

        if len(alleles) >= 2:
            ref = alleles[0]
            alt = ",".join(alleles[1:])

    rows.append({
        "rsID": rsid,
        "CHROM": chrom,
        "POS": pos,
        "REF": ref,
        "ALT": alt,
    })

df = pd.DataFrame(
    rows,
    columns=["rsID", "CHROM", "POS", "REF", "ALT"]
)

df["POS"] = df["POS"].astype("Int64")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"Results saved to: {OUTPUT_FILE}")
print(f"Total SNPs: {len(df)}")
print(f"Successfully mapped to GRCh38: {df['POS'].notna().sum()}")
print(f"Failed mappings: {df['POS'].isna().sum()}")
