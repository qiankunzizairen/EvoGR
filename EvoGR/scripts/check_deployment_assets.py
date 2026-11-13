#!/usr/bin/env python3
"""Server-side preflight; intentionally does not run Evo2 extraction."""
from __future__ import annotations
import argparse, json, os, platform, sys
from pathlib import Path
import _bootstrap  # noqa: F401
from aim_evo2.config import load_config, resolve_path
from aim_evo2.data import expected_n_aim, load_validated_data
from aim_evo2.utils import sha256_file

def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight configured Linux Evo2 deployment assets.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        data = load_validated_data(config)
        required = [resolve_path(config, config["evo2"]["checkpoint_path"])]
        required += [resolve_path(config, config["data"][key]) for key in ("genotype_csv", "label_metadata_tsv", "allele_metadata_csv", "reference_fasta", "reference_fai")]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ValueError("missing files: " + ", ".join(missing))
        model_manifest = Path(config["_root"]) / "models/evo2_7b_base_model_manifest.json"
        if model_manifest.is_file():
            expected = json.loads(model_manifest.read_text(encoding="utf-8")).get("checkpoint_sha256")
            if expected and sha256_file(required[0]) != expected:
                raise ValueError("checkpoint_sha256 mismatch")
        try:
            import torch
            torch_info = f"torch={torch.__version__} cuda={int(torch.cuda.is_available())}"
        except ImportError as exc:
            raise ValueError("torch is missing; install the Linux Evo2 environment") from exc
        try:
            import evo2  # noqa: F401
        except ImportError as exc:
            raise ValueError("evo2 is missing; run this preflight in the Linux Evo2 environment") from exc
        print(f"preflight OK: platform={platform.system()} python={sys.version_info.major}.{sys.version_info.minor} x={config['data']['added_aims_per_autosome']} n_aim={data.n_aim}/{expected_n_aim(config)} {torch_info} offline={os.environ.get('HF_HUB_OFFLINE')}")
    except (KeyError, OSError, ValueError) as exc:
        print(f"preflight FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
