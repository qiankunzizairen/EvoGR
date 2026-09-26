#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401
from aim_evo2.utils import sha256_file, utc_now_iso, write_json


def download(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    info = api.model_info(args.repo_id)
    revision = info.sha
    output_dir = Path(args.output_dir)
    snapshot_download(
        repo_id=args.repo_id,
        revision=revision,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
    )
    files = sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file())
    write_json(
        Path(args.manifest),
        {
            "repo_id": args.repo_id,
            "revision": revision,
            "download_time_utc": utc_now_iso(),
            "source_files": files,
        },
    )
    print(f"Downloaded {args.repo_id}@{revision} to {output_dir}")


def _part_index(path: Path) -> int:
    suffix = path.name.rsplit(".part", 1)[-1]
    try:
        return int(suffix)
    except ValueError as exc:
        raise ValueError(f"Invalid shard name {path.name}") from exc


def merge(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    output_path = Path(args.output_path)
    full_candidates = list(source_dir.rglob("evo2_7b_base.pt"))
    revision = None
    download_manifest = Path(args.download_manifest)
    if download_manifest.exists():
        import json

        revision = json.loads(download_manifest.read_text(encoding="utf-8")).get("revision")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if full_candidates:
        shutil.copy2(full_candidates[0], output_path)
    else:
        parts = sorted(source_dir.rglob("evo2_7b_base.pt.part*"), key=_part_index)
        if not parts:
            raise FileNotFoundError(f"No evo2_7b_base.pt or .part shards found under {source_dir}")
        indices = [_part_index(path) for path in parts]
        expected = list(range(len(parts)))
        if indices != expected:
            raise ValueError(f"Shard indices must be consecutive from 0; found {indices}")
        with output_path.open("wb") as out:
            for part in parts:
                with part.open("rb") as handle:
                    shutil.copyfileobj(handle, out)
    write_json(
        Path(args.manifest),
        {
            "model_name": "evo2_7b_base",
            "repo_id": args.repo_id,
            "revision": revision,
            "checkpoint_path": str(output_path),
            "checkpoint_size_bytes": output_path.stat().st_size,
            "checkpoint_sha256": sha256_file(output_path),
        },
    )
    print(f"Merged checkpoint: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or merge the Evo2 7B base checkpoint without loading it.")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download")
    d.add_argument("--repo-id", default="arcinstitute/evo2_7b_base")
    d.add_argument("--output-dir", default="models/evo2_7b_base_hf")
    d.add_argument("--manifest", default="models/evo2_7b_base_download_manifest.json")
    d.set_defaults(func=download)
    m = sub.add_parser("merge")
    m.add_argument("--repo-id", default="arcinstitute/evo2_7b_base")
    m.add_argument("--source-dir", default="models/evo2_7b_base_hf")
    m.add_argument("--output-path", default="models/evo2_7b_base.pt")
    m.add_argument("--download-manifest", default="models/evo2_7b_base_download_manifest.json")
    m.add_argument("--manifest", default="models/evo2_7b_base_model_manifest.json")
    m.set_defaults(func=merge)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
