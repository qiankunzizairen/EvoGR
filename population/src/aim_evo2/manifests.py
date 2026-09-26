from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import all_context_lengths, all_flank_bp, resolve_path
from .data import ValidatedData
from .utils import sha256_file, sha256_text


def rsid_order_sha256(rsid_order: list[str]) -> str:
    """Hash an ordered rsID vector without conflating different boundaries."""
    return sha256_text("".join(f"{len(rsid)}:{rsid}\n" for rsid in rsid_order))


def embedding_manifest_contract(config: dict, data: ValidatedData) -> dict[str, Any]:
    root = Path(config["_root"])
    sequence_dir = resolve_path(config, config["artifacts"]["sequence_dir"])
    return {
        "schema_version": "2.0",
        "model_name": config["evo2"]["model_name"],
        "repo_id": config["evo2"]["repo_id"],
        "checkpoint_sha256": sha256_file(resolve_path(config, config["evo2"]["checkpoint_path"])),
        "layer_name": config["evo2"]["layer_name"],
        "pool_radii": [int(value) for value in config["evo2"]["pool_radii"]],
        "flank_bp": all_flank_bp(config),
        "context_lengths": all_context_lengths(config),
        "genotype_sha256": sha256_file(resolve_path(config, config["data"]["genotype_csv"])),
        "allele_metadata_sha256": sha256_file(resolve_path(config, config["data"]["allele_metadata_csv"])),
        "reference_fasta_sha256": sha256_file(resolve_path(config, config["data"]["reference_fasta"])),
        "reference_fai_sha256": sha256_file(resolve_path(config, config["data"]["reference_fai"])),
        "rsid_order_sha256": rsid_order_sha256(data.rsid_order),
        "sequence_manifest_sha256": sha256_file(sequence_dir / "aim_allele_manifest.tsv"),
        "x": int(config["data"]["added_aims_per_autosome"]),
        "n_aim": data.n_aim,
        "dtype_saved": "float32",
    }


def representation_manifest_contract(
    config: dict,
    data: ValidatedData,
    embedding_path: Path,
    hidden_dim: int,
) -> dict[str, Any]:
    lengths = all_context_lengths(config)
    shape = [len(data.genotypes), data.n_aim, 2 * int(hidden_dim)]
    return {
        "schema_version": "2.0",
        "embedding_sha256": sha256_file(embedding_path),
        "genotype_sha256": sha256_file(resolve_path(config, config["data"]["genotype_csv"])),
        "rsid_order_sha256": rsid_order_sha256(data.rsid_order),
        "context_lengths": lengths,
        "n_samples": len(data.genotypes),
        "n_aim": data.n_aim,
        "hidden_dim": int(hidden_dim),
        "dtype": "float16",
        "shapes": {str(length): shape for length in lengths},
    }


def manifest_mismatches(actual: Any, expected: Any, prefix: str = "") -> list[str]:
    """Return precise contract mismatches; extra actual fields are permitted."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{prefix or '<root>'}: expected mapping, found {type(actual).__name__}"]
        mismatches: list[str] = []
        for key, expected_value in expected.items():
            path = f"{prefix}.{key}" if prefix else key
            if key not in actual:
                mismatches.append(f"{path}: missing")
            else:
                mismatches.extend(manifest_mismatches(actual[key], expected_value, path))
        return mismatches
    if actual != expected:
        return [f"{prefix}: expected {expected!r}, found {actual!r}"]
    return []


def assert_manifest_current(path: Path, expected: dict[str, Any], artifact_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{artifact_name} manifest missing: {path}")
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{artifact_name} manifest unreadable: {path}: {exc}") from exc
    mismatches = manifest_mismatches(actual, expected)
    if mismatches:
        detail = "\n".join(f"- {item}" for item in mismatches)
        raise ValueError(f"stale {artifact_name} cache; manifest mismatch:\n{detail}")
    return actual


def is_manifest_stale(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Pure stale predicate used by cache consumers and tests."""
    return bool(manifest_mismatches(actual, expected))


def stale_reasons(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    return manifest_mismatches(actual, expected)
