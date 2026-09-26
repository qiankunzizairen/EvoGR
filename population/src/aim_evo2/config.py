from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_VARIABLE_FLANK_BP = (10, 50, 100, 500, 1000)


def context_length_from_flank(flank_bp: int) -> int:
    flank = int(flank_bp)
    if flank < 0:
        raise ValueError(f"flank_bp must be non-negative, found {flank}")
    return 2 * flank + 1


def fixed_flank_bp(config: dict[str, Any]) -> int:
    return int(config["sequence"]["fixed_flank_bp"])


def fixed_context_length(config: dict[str, Any]) -> int:
    return context_length_from_flank(fixed_flank_bp(config))


def variable_flank_bp(config: dict[str, Any], phase: str) -> list[int]:
    if phase not in {"train", "eval"}:
        raise ValueError(f"phase must be train or eval, found {phase!r}")
    return [int(value) for value in config["variable_context"][f"{phase}_flank_bp"]]


def variable_context_lengths(config: dict[str, Any], phase: str) -> list[int]:
    return [context_length_from_flank(value) for value in variable_flank_bp(config, phase)]


def all_flank_bp(config: dict[str, Any]) -> list[int]:
    return sorted({fixed_flank_bp(config), *variable_flank_bp(config, "train"), *variable_flank_bp(config, "eval")})


def all_context_lengths(config: dict[str, Any]) -> list[int]:
    return [context_length_from_flank(value) for value in all_flank_bp(config)]


def load_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping")
    root = config_path.resolve().parent.parent
    config["_config_path"] = str(config_path)
    config["_root"] = str(root)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if int(config["model"]["num_classes"]) != 5 or int(config["cv"]["n_splits"]) != 5:
        raise ValueError("This project requires exactly 5 classes and 5 outer folds")
    data = config["data"]
    if int(data["base_aim_count"]) <= 0 or int(data["autosome_count"]) <= 0 or int(data["added_aims_per_autosome"]) < 0:
        raise ValueError("added_aims_per_autosome must be non-negative")
    if config["evo2"]["model_name"] != "evo2_7b_base":
        raise ValueError("evo2.model_name must be evo2_7b_base")
    if fixed_flank_bp(config) != 50:
        raise ValueError("sequence.fixed_flank_bp must be 50")
    for phase in ("train", "eval"):
        flanks = variable_flank_bp(config, phase)
        if tuple(flanks) != REQUIRED_VARIABLE_FLANK_BP:
            raise ValueError(
                f"variable_context.{phase}_flank_bp must be {list(REQUIRED_VARIABLE_FLANK_BP)}"
            )
    maximum = int(config["sequence"]["max_context_length"])
    if maximum != 8192:
        raise ValueError("sequence.max_context_length must be 8192 for evo2_7b_base")
    radii = [int(x) for x in config["evo2"]["pool_radii"]]
    if not radii or any(radius < 0 for radius in radii) or len(radii) != len(set(radii)):
        raise ValueError("evo2.pool_radii must be unique non-negative integers")
    if int(config["evo2"]["active_pool_radius"]) not in radii:
        raise ValueError("evo2.active_pool_radius must be listed in evo2.pool_radii")
    for flank, length in zip(all_flank_bp(config), all_context_lengths(config), strict=True):
        if length > maximum:
            raise ValueError(f"context length {length} exceeds configured maximum {maximum}")
        if max(radii) > flank:
            raise ValueError(f"pool radius {max(radii)} exceeds flank boundary {flank} for L={length}")
    if len(config.get("results", {})) != 3 or set(config["results"]) != {"fixed_full_json", "variable_snp_json", "variable_context_json"}:
        raise ValueError("results must define exactly three formal JSON paths")


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_root"]) / path


def save_resolved_config(config: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    output.write_text(yaml.safe_dump(clean, sort_keys=False), encoding="utf-8")
