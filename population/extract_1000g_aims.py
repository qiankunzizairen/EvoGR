import argparse
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import csv
import re
import shutil
import subprocess
import tempfile
import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor


# Extraction profiles
AIM56_INPUT_FILE = Path("data/aims56_grch38.csv")
AIM56_OUTPUT_FILE = Path("data/aims56_1000g.vcf")
CHN100K_INPUT_DIR = Path("data/CHN100K_aims")
CHN100K_OUTPUT_FILE = Path("data/CHN100K_aims_1000g.vcf")

# 1000 Genomes phased VCF directory
BASE_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/"
    "data_collections/1000G_2504_high_coverage/working/"
    "20220422_3202_phased_SNV_INDEL_SV"
)

# Remote request settings
MAX_RETRIES = 1
RETRY_DELAY_SECONDS = 5
COMMAND_TIMEOUT_SECONDS = 900
CURL_RETRIES = 5
RANGE_PROXY_RETRIES = 3
RANGE_CHUNK_SIZE = 262144
RANGE_REQUEST_RETRIES = 5

# Supported chromosomes in this 1000G phased panel
SUPPORTED_CHROMS = {str(i) for i in range(1, 23)} | {"X"}


def check_bcftools():
    """Check whether bcftools is available in PATH."""
    bcftools_path = shutil.which("bcftools")

    if bcftools_path is None:
        raise RuntimeError(
            "bcftools was not found in PATH. "
            "Please install bcftools before running this script."
        )

    result = subprocess.run(
        ["bcftools", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )

    version_line = result.stdout.splitlines()[0]
    print(f"Using {version_line}")


def get_remote_size(url):
    result = subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--head",
            "--http1.1",
            "--retry",
            str(CURL_RETRIES),
            "--retry-all-errors",
            url,
        ],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=COMMAND_TIMEOUT_SECONDS,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to read remote file metadata: {result.stderr.strip()}"
        )

    content_lengths = re.findall(
        r"^content-length:\s*(\d+)\s*$",
        result.stdout,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    if not content_lengths:
        raise RuntimeError(f"Remote file size was not reported for {url}")

    return int(content_lengths[-1])


@contextmanager
def curl_range_proxy(remote_url):
    remote_files = {
        "/source.vcf.gz": remote_url,
        "/source.vcf.gz.tbi": remote_url + ".tbi",
    }
    remote_sizes = {
        path: get_remote_size(url)
        for path, url in remote_files.items()
    }
    index_directory = tempfile.TemporaryDirectory(
        prefix="aims56_index_"
    )
    index_path = Path(index_directory.name) / "source.vcf.gz.tbi"
    index_result = subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--http1.1",
            "--retry",
            str(CURL_RETRIES),
            "--retry-all-errors",
            "--output",
            str(index_path),
            remote_url + ".tbi",
        ],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=COMMAND_TIMEOUT_SECONDS,
    )

    if index_result.returncode != 0:
        index_directory.cleanup()
        raise RuntimeError(
            f"Failed to download the remote VCF index: "
            f"{index_result.stderr.strip()}"
        )

    class RangeRequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def send_not_found(self):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()

        def do_HEAD(self):
            if self.path not in remote_files:
                self.send_not_found()
                return

            self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header(
                "Content-Length",
                str(remote_sizes[self.path]),
            )
            self.send_header("Connection", "close")
            self.end_headers()

        def do_GET(self):
            if self.path not in remote_files:
                self.send_not_found()
                return

            remote_size = remote_sizes[self.path]
            range_header = self.headers.get("Range")

            if not range_header and self.path.endswith(".vcf.gz"):
                process = subprocess.Popen(
                    [
                        "curl",
                        "--fail",
                        "--silent",
                        "--show-error",
                        "--location",
                        "--http1.1",
                        "--retry",
                        str(CURL_RETRIES),
                        "--retry-all-errors",
                        remote_files[self.path],
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                self.send_response(200)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(remote_size))
                self.send_header("Connection", "close")
                self.end_headers()

                try:
                    while True:
                        chunk = process.stdout.read(65536)

                        if not chunk:
                            break

                        self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    process.terminate()
                    process.wait()

                return

            start = 0
            end = remote_size - 1

            if range_header:
                match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header)

                if match is None:
                    self.send_error(416)
                    return

                start = int(match.group(1))

                if match.group(2):
                    end = min(int(match.group(2)), remote_size - 1)
                else:
                    end = min(start + RANGE_CHUNK_SIZE - 1, remote_size - 1)

            expected_length = end - start + 1
            result = None
            for request_attempt in range(1, RANGE_REQUEST_RETRIES + 1):
                result = subprocess.run(
                    [
                    "curl",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--location",
                    "--http1.1",
                    "--retry",
                    str(CURL_RETRIES),
                    "--retry-all-errors",
                    "--range",
                    f"{start}-{end}",
                        remote_files[self.path],
                    ],
                    capture_output=True,
                    timeout=COMMAND_TIMEOUT_SECONDS,
                )
                if result.returncode == 0 and len(result.stdout) == expected_length:
                    break
                if request_attempt < RANGE_REQUEST_RETRIES:
                    time.sleep(min(request_attempt, 3))

            if result.returncode != 0:
                self.send_error(502)
                return

            data = result.stdout
            actual_end = start + len(data) - 1
            self.send_response(206 if range_header else 200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(data)))

            if range_header:
                self.send_header(
                    "Content-Range",
                    f"bytes {start}-{actual_end}/{remote_size}",
                )

            self.send_header("Connection", "close")
            self.end_headers()

            try:
                self.wfile.write(data)
            except BrokenPipeError:
                pass

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        RangeRequestHandler,
    )
    server.daemon_threads = True
    server_thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )
    server_thread.start()

    try:
        host, port = server.server_address
        yield (
            f"http://{host}:{port}/source.vcf.gz",
            index_path,
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()
        index_directory.cleanup()


def normalize_chrom(chrom):
    """Normalize chromosome names to 1-22 or X."""
    chrom = str(chrom).strip()

    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]

    chrom = chrom.upper()

    if chrom not in SUPPORTED_CHROMS:
        raise ValueError(
            f"Unsupported chromosome '{chrom}'. "
            "This script supports chromosomes 1-22 and X."
        )

    return chrom


def chromosome_sort_key(chrom):
    """Return a natural chromosome sorting key."""
    if chrom == "X":
        return 23

    return int(chrom)


def get_remote_vcf_url(chrom):
    """Build the remote 1000G VCF URL for a chromosome."""
    if chrom == "X":
        filename = (
            "1kGP_high_coverage_Illumina.chrX."
            "filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
        )
    else:
        filename = (
            f"1kGP_high_coverage_Illumina.chr{chrom}."
            "filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
        )

    return f"{BASE_URL}/{filename}"


def _target_from_row(row, source_path, line_number):
    """Normalize one target-SNP row from a CSV or TSV source."""
    rsid = (row.get("rsID") or "").strip()
    if not rsid:
        raise ValueError(f"{source_path}:{line_number}: empty rsID")
    try:
        pos = int((row.get("POS") or "").strip())
    except ValueError as exc:
        raise ValueError(
            f"{source_path}:{line_number}: invalid POS value for {rsid}: {row.get('POS')!r}"
        ) from exc
    if pos <= 0:
        raise ValueError(f"{source_path}:{line_number}: POS must be positive for {rsid}")
    ref = (row.get("REF") or "").strip()
    alt = (row.get("ALT") or "").strip()
    if not ref or not alt:
        raise ValueError(f"{source_path}:{line_number}: empty REF or ALT for {rsid}")
    return {
        "rsID": rsid,
        "CHROM": normalize_chrom(row.get("CHROM") or ""),
        "POS": pos,
        "REF": ref,
        "ALT": alt,
    }


def _load_target_file(path, delimiter):
    """Load one target-SNP table with the required leading schema."""
    required_columns = {"rsID", "CHROM", "POS", "REF", "ALT"}

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    targets = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)

        if reader.fieldnames is None:
            raise ValueError(f"{path}: input table has no header.")

        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                f"{path}: missing required columns: {sorted(missing_columns)}"
            )

        for line_number, row in enumerate(reader, start=2):
            targets.append(_target_from_row(row, path, line_number))

    return targets


def _validate_unique_targets(targets, source_label):
    """Reject duplicate rsIDs or genomic coordinates across all input tables."""
    seen_rsids = set()
    seen_coordinates = set()
    for target in targets:
        rsid = target["rsID"]
        coordinate = (target["CHROM"], target["POS"])
        if rsid in seen_rsids:
            raise ValueError(f"{source_label}: duplicate rsID found: {rsid}")
        if coordinate in seen_coordinates:
            raise ValueError(
                f"{source_label}: duplicate target coordinate: "
                f"chr{coordinate[0]}:{coordinate[1]}"
            )
        seen_rsids.add(rsid)
        seen_coordinates.add(coordinate)


def load_target_snps(csv_path):
    """Load the original AIM56 targets from the GRCh38 CSV file."""
    targets = _load_target_file(csv_path, ",")
    _validate_unique_targets(targets, str(csv_path))

    if not targets:
        raise ValueError(f"{csv_path}: no SNP records were found.")

    print(f"Loaded {len(targets)} target SNPs")

    return targets


def load_chn100k_target_snps(input_dir):
    """Load all chromosome candidate TSVs from the CHN100K AIM directory."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"CHN100K AIM directory not found: {input_dir}")

    files_by_chrom = {}
    filename_pattern = re.compile(r"^chr([0-9]+)_top200\.tsv$")
    for path in input_dir.glob("*.tsv"):
        match = filename_pattern.fullmatch(path.name)
        if match is None:
            continue
        chrom = normalize_chrom(match.group(1))
        if chrom in files_by_chrom:
            raise ValueError(f"{input_dir}: multiple candidate TSVs for chromosome {chrom}")
        files_by_chrom[chrom] = path

    if not files_by_chrom:
        raise ValueError(f"{input_dir}: no chr*_top200.tsv candidate files were found")

    targets = []
    for chrom in sorted(files_by_chrom, key=chromosome_sort_key):
        path = files_by_chrom[chrom]
        file_targets = _load_target_file(path, "\t")
        mismatched = [
            target for target in file_targets if target["CHROM"] != chrom
        ]
        if mismatched:
            first = mismatched[0]
            raise ValueError(
                f"{path}: filename chromosome {chrom} disagrees with "
                f"{first['rsID']} chromosome {first['CHROM']}"
            )
        targets.extend(file_targets)

    _validate_unique_targets(targets, str(input_dir))
    print(f"Loaded {len(targets)} CHN100K target SNPs from {len(files_by_chrom)} TSV files")
    return targets


def group_targets_by_chromosome(targets):
    """Group target SNPs by chromosome."""
    grouped = defaultdict(list)

    for target in targets:
        grouped[target["CHROM"]].append(target)

    for chrom in grouped:
        grouped[chrom].sort(key=lambda item: item["POS"])

    return grouped


def build_region_string(chrom, targets):
    """Build a bcftools region string for one chromosome."""
    regions = []

    for target in targets:
        pos = target["POS"]

        # The 1000G GRCh38 files use UCSC-style chromosome names.
        regions.append(f"chr{chrom}:{pos}-{pos}")

    return ",".join(regions)


def run_bcftools_extract(chrom, targets, output_path, local_vcf_dir=None):
    """Extract target positions from one remote chromosome VCF."""
    remote_url = get_remote_vcf_url(chrom)
    region_string = build_region_string(chrom, targets)

    source = remote_url
    if local_vcf_dir is not None:
        local_vcf = local_vcf_dir / Path(remote_url).name
        local_index = Path(f"{local_vcf}.tbi")
        if not local_vcf.is_file() or not local_index.is_file():
            raise FileNotFoundError(
                f"Missing local VCF or index for chr{chrom}: "
                f"{local_vcf} and {local_index}"
            )
        source = str(local_vcf)

    command = [
        "bcftools",
        "view",
        "--no-version",
        "--types",
        "snps",
        "--regions",
        region_string,
        "--output-type",
        "v",
        "--output",
        str(output_path),
        source,
    ]

    print(
        f"Extracting {len(targets)} target SNPs "
        f"from chromosome {chrom}"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=COMMAND_TIMEOUT_SECONDS,
            )

            if result.returncode == 0:
                return

            error_message = result.stderr.strip()

        except subprocess.TimeoutExpired:
            error_message = (
                f"The request exceeded "
                f"{COMMAND_TIMEOUT_SECONDS} seconds."
            )

        print(
            f"Attempt {attempt}/{MAX_RETRIES} failed "
            f"for chromosome {chrom}: {error_message}"
        )

        if attempt < MAX_RETRIES:
            print(
                f"Retrying chromosome {chrom} "
                f"in {RETRY_DELAY_SECONDS} seconds"
            )
            time.sleep(RETRY_DELAY_SECONDS)

    print(
        f"Direct remote access failed for chromosome {chrom}; "
        "switching to curl range transfer"
    )

    with curl_range_proxy(remote_url) as (proxy_url, index_path):
        fallback_command = command.copy()
        fallback_command[-1] = (
            f"{proxy_url}##idx##file://{index_path}"
        )

        for attempt in range(1, RANGE_PROXY_RETRIES + 1):
            result = subprocess.run(
                fallback_command,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=COMMAND_TIMEOUT_SECONDS,
            )

            if result.returncode == 0:
                return

            print(
                f"Range transfer attempt {attempt}/"
                f"{RANGE_PROXY_RETRIES} failed for chromosome {chrom}: "
                f"{result.stderr.strip()}"
            )

    raise RuntimeError(
        f"Failed to extract chromosome {chrom} "
        "using both direct and curl range transfer methods."
    )


def merge_vcf_files(vcf_files, output_path):
    """Merge chromosome-specific VCF files into one VCF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_output = output_path.with_suffix(".vcf.tmp")

    reference_column_header = None
    header_written = False

    with temporary_output.open("w", encoding="utf-8") as output_handle:

        for vcf_path in vcf_files:
            with vcf_path.open("r", encoding="utf-8") as input_handle:

                for line in input_handle:

                    if line.startswith("#CHROM"):
                        current_column_header = line

                        if reference_column_header is None:
                            reference_column_header = current_column_header
                        elif current_column_header != reference_column_header:
                            raise RuntimeError(
                                "Sample columns differ between chromosome VCF files."
                            )

                    if line.startswith("#"):
                        if not header_written:
                            output_handle.write(line)
                        continue

                    output_handle.write(line)

            header_written = True

    temporary_output.replace(output_path)


def inspect_output_vcf(output_path, targets):
    """Check whether the requested SNP coordinates are present."""
    target_by_coordinate = defaultdict(list)

    for target in targets:
        key = (target["CHROM"], target["POS"])
        target_by_coordinate[key].append(target)

    found_coordinates = set()
    found_rsids = set()
    allele_mismatches = []
    record_count = 0

    with output_path.open("r", encoding="utf-8") as handle:

        for line in handle:
            if line.lstrip().startswith("#") or not line.strip():
                continue

            fields = line.rstrip("\n").split("\t")

            if len(fields) < 5:
                continue

            chrom = normalize_chrom(fields[0])
            pos = int(fields[1])
            variant_id = fields[2]

            key = (chrom, pos)

            if key in target_by_coordinate:
                found_coordinates.add(key)

                expected_alleles = {(t["REF"].upper(), t["ALT"].upper()) for t in target_by_coordinate[key]}
                observed_alleles = (fields[3].upper(), fields[4].split(",")[0].upper())
                if observed_alleles not in expected_alleles:
                    allele_mismatches.append((key, expected_alleles, observed_alleles))

                record_ids = set(variant_id.split(";"))

                for expected_rsid in (t["rsID"] for t in target_by_coordinate[key]):
                    if expected_rsid in record_ids:
                        found_rsids.add(expected_rsid)

            record_count += 1

    missing_coordinates = []

    for target in targets:
        key = (target["CHROM"], target["POS"])

        if key not in found_coordinates:
            missing_coordinates.append(target)

    print(f"Output VCF records: {record_count}")
    print(
        f"Target coordinates found: "
        f"{len(found_coordinates)}/{len(targets)}"
    )
    print(
        f"Target rsIDs confirmed: "
        f"{len(found_rsids)}/{len(targets)}"
    )

    if missing_coordinates:
        print("The following target coordinates were not found:")

        for target in missing_coordinates:
            print(
                f"# {target['rsID']} "
                f"chr{target['CHROM']}:{target['POS']}"
            )
    else:
        print("All target coordinates were found in the 1000G VCF panel.")

    if allele_mismatches:
        print(f"Target allele mismatches: {len(allele_mismatches)}")

    missing_rsids = [
        target
        for target in targets
        if target["rsID"] not in found_rsids
    ]

    if missing_rsids:
        print(
            "Some coordinates were retrieved but their expected rsIDs are not "
            "present in the VCF ID field; coordinate validation remains authoritative."
        )

    return {
        "record_count": record_count,
        "found_coordinate_count": len(found_coordinates),
        "missing_coordinates": missing_coordinates,
        "allele_mismatches": allele_mismatches,
        "found_rsid_count": len(found_rsids),
    }


def read_vcf_sample_columns(path):
    """Return ordered VCF sample IDs from the #CHROM header."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.lstrip().startswith("#CHROM"):
                fields = line.lstrip().rstrip("\n").split("\t")
                if len(fields) < 10:
                    raise ValueError(f"{path}:{line_number}: VCF has no sample columns")
                return fields[9:]
    raise ValueError(f"{path}: #CHROM header not found")


def validate_sample_columns_match(output_path, reference_vcf):
    """Require the extracted VCF sample columns to match the AIM56 VCF exactly."""
    if not reference_vcf.exists():
        raise FileNotFoundError(
            f"Reference VCF for sample comparison not found: {reference_vcf}"
        )
    output_samples = read_vcf_sample_columns(output_path)
    reference_samples = read_vcf_sample_columns(reference_vcf)
    if output_samples != reference_samples:
        raise RuntimeError(
            "Extracted VCF sample columns differ from the AIM56 VCF reference. "
            f"output_count={len(output_samples)} reference_count={len(reference_samples)}"
        )
    print(
        f"Sample columns match {reference_vcf}: {len(output_samples)} samples"
    )


def extract_target_vcf(
    targets,
    output_path,
    temporary_prefix,
    sample_reference_vcf=None,
    local_vcf_dir=None,
):
    """Run the shared chromosome extraction pipeline for one target collection."""
    grouped_targets = group_targets_by_chromosome(targets)
    chromosomes = sorted(grouped_targets.keys(), key=chromosome_sort_key)
    print("Target chromosomes: " + ", ".join(chromosomes))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=temporary_prefix,
        dir=output_path.parent,
    ) as temporary_directory:
        temporary_directory = Path(temporary_directory)
        # Fetch chromosome files concurrently; merge remains in natural
        # chromosome order so output is deterministic and byte-format stable.
        chromosome_vcfs = [temporary_directory / f"chr{chrom}.vcf" for chrom in chromosomes]
        with ThreadPoolExecutor(max_workers=1) as pool:
            futures = [
                pool.submit(
                    run_bcftools_extract,
                    chrom,
                    grouped_targets[chrom],
                    out,
                    local_vcf_dir,
                )
                for chrom, out in zip(chromosomes, chromosome_vcfs)
            ]
            for future in futures:
                future.result()

        staged_output = temporary_directory / output_path.name
        print("Merging chromosome-specific VCF files")
        merge_vcf_files(chromosome_vcfs, staged_output)
        inspection = inspect_output_vcf(staged_output, targets)
        if inspection["missing_coordinates"]:
            raise RuntimeError(
                f"Refusing to write incomplete output: "
                f"{len(inspection['missing_coordinates'])}/{len(targets)} target coordinates are missing"
            )
        # The downloaded 1000G VCF is authoritative for REF/ALT. Candidate
        # tables can use a different allele representation at the same locus;
        # retain the VCF records and report those differences as warnings.
        if sample_reference_vcf is not None:
            validate_sample_columns_match(staged_output, sample_reference_vcf)
        staged_output.replace(output_path)

    print(f"Saved output VCF to {output_path}")


def parse_args(argv=None):
    """Parse the AIM56 and CHN100K extraction entry points."""
    parser = argparse.ArgumentParser(
        description="Extract AIM SNPs from the 1000 Genomes phased GRCh38 VCF panel."
    )
    subparsers = parser.add_subparsers(dest="mode")
    aim56 = subparsers.add_parser("aim56", help="extract the existing 56 AIM SNPs")
    aim56.add_argument("--input", type=Path, default=AIM56_INPUT_FILE)
    aim56.add_argument("--output", type=Path, default=AIM56_OUTPUT_FILE)
    aim56.add_argument(
        "--vcf-dir",
        type=Path,
        default=None,
        help="use downloaded local chromosome VCFs instead of remote URLs",
    )
    chn100k = subparsers.add_parser(
        "chn100k", help="extract all CHN100K AIM candidate SNPs"
    )
    chn100k.add_argument("--input-dir", type=Path, default=CHN100K_INPUT_DIR)
    chn100k.add_argument("--output", type=Path, default=CHN100K_OUTPUT_FILE)
    chn100k.add_argument(
        "--sample-reference-vcf", type=Path, default=AIM56_OUTPUT_FILE,
        help="VCF whose ordered sample columns must match the CHN100K output",
    )
    chn100k.add_argument(
        "--vcf-dir",
        type=Path,
        default=None,
        help="use downloaded local chromosome VCFs instead of remote URLs",
    )

    if argv is None and len(sys.argv) == 1:
        argv = ["aim56"]
    args = parser.parse_args(argv)
    if args.mode is None:
        parser.error("choose an extraction mode: aim56 or chn100k")
    return args


def main():
    """Run one of the two 1000G AIM extraction entry points."""
    args = parse_args()
    check_bcftools()
    if args.mode == "aim56":
        targets = load_target_snps(args.input)
        extract_target_vcf(
            targets,
            args.output,
            "aims56_1000g_",
            local_vcf_dir=args.vcf_dir,
        )
    else:
        targets = load_chn100k_target_snps(args.input_dir)
        extract_target_vcf(
            targets,
            args.output,
            "chn100k_aims_",
            args.sample_reference_vcf,
            args.vcf_dir,
        )


if __name__ == "__main__":
    main()
