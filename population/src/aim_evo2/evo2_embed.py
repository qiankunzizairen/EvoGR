from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .utils import sha256_file, utc_now_iso, write_json
from .config import all_context_lengths
from .data import load_validated_data
from .manifests import embedding_manifest_contract


def read_fasta_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_id: str | None = None
    chunks: list[str] = []
    with path.open(encoding="ascii") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(chunks)))
                current_id = line[1:]
                chunks = []
            else:
                chunks.append(line.upper())
    if current_id is not None:
        records.append((current_id, "".join(chunks)))
    return records


def _available_block_names(model: torch.nn.Module) -> list[str]:
    return [name for name, _ in model.named_modules() if "blocks." in name]


def _encode_sequence(model: Any, sequence: str) -> torch.Tensor:
    try:
        device = next(model.model.parameters()).device
    except StopIteration:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    token_ids = model.tokenizer.tokenize(sequence)
    return torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)


def _layer_embedding(outputs: Any, layer_name: str) -> torch.Tensor:
    if isinstance(outputs, tuple) and len(outputs) == 2:
        _, embeddings = outputs
    elif isinstance(outputs, dict) and "embeddings" in outputs:
        embeddings = outputs["embeddings"]
    else:
        raise TypeError(f"Unexpected Evo2 output type: {type(outputs).__name__}")
    if layer_name not in embeddings:
        raise KeyError(f"Evo2 did not return the requested layer embedding: {layer_name}")
    return embeddings[layer_name]


def extract_embeddings(config: dict) -> tuple[Path, Path]:
    from evo2 import Evo2

    root = Path(config["_root"])
    sequence_manifest = root / config["artifacts"]["sequence_dir"] / "aim_allele_manifest.tsv"
    fasta_path = root / config["artifacts"]["sequence_dir"] / "aim_alleles.fasta"
    checkpoint_path = root / config["evo2"]["checkpoint_path"]
    output_dir = root / config["artifacts"]["embedding_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_name = config["evo2"]["layer_name"]
    model = Evo2(config["evo2"]["model_name"], local_path=str(checkpoint_path))
    model.model.eval()
    names = dict(model.model.named_modules())
    if layer_name not in names:
        raise ValueError(f"Layer {layer_name!r} not found. Available block modules: {_available_block_names(model.model)[:50]}")
    data = load_validated_data(config)
    manifest = pd.read_csv(sequence_manifest, sep="\t")
    expected_lengths = all_context_lengths(config)
    observed_lengths = sorted(manifest["context_length"].astype(int).unique()) if "context_length" in manifest else []
    if observed_lengths != expected_lengths:
        raise ValueError(f"sequence manifest context lengths mismatch: expected {expected_lengths}, found {observed_lengths}")
    records = dict(read_fasta_records(fasta_path))
    pool_radii = [int(radius) for radius in config["evo2"]["pool_radii"]]
    pooled_by_length: dict[str, dict[str, dict[str, dict[int, torch.Tensor]]]] = {}
    if "context_length" not in manifest:
        raise ValueError("sequence manifest lacks required context_length column")
    context_lengths = sorted({int(x) for x in manifest["context_length"]})
    hidden_dim: int | None = None
    with torch.inference_mode():
        for _, row in manifest.iterrows():
            seq_id = row["seq_id"]
            rsid = row["rsID"]
            allele_index = int(row["allele_index"])
            sequence = records[seq_id]
            context_length = int(row.get("context_length", len(sequence)))
            center = context_length // 2
            input_ids = _encode_sequence(model, sequence)
            if input_ids.shape[1] != context_length:
                raise ValueError(f"{seq_id}: tokenizer produced {input_ids.shape[1]} tokens for sequence length {context_length}")
            outputs = model(input_ids, return_embeddings=True, layer_names=[layer_name])
            hidden = _layer_embedding(outputs, layer_name)
            if hidden.ndim == 2:
                hidden = hidden.unsqueeze(0)
            if hidden.shape[1] != context_length:
                raise ValueError(f"{seq_id}: Evo2 hidden length {hidden.shape[1]} != input length {context_length}")
            hidden_dim = int(hidden.shape[-1])
            length_cache = pooled_by_length.setdefault(str(context_length), {str(radius): {} for radius in pool_radii})
            for radius in pool_radii:
                start = center - radius
                end = center + radius + 1
                pooled = hidden[:, start:end, :].mean(dim=1).squeeze(0).detach().cpu().float()
                length_cache[str(radius)].setdefault(rsid, {})[allele_index] = pooled
    embedding_path = output_dir / "allele_embeddings.pt"
            # Store new schema under context length while retaining radius keys.
    cache = pooled_by_length[str(context_lengths[0])] if len(context_lengths) == 1 else pooled_by_length
    torch.save(cache, embedding_path)
    manifest_path = output_dir / "embedding_manifest.json"
    payload = embedding_manifest_contract(config, data)
    payload.update({"hidden_dim": hidden_dim, "created_at_utc": utc_now_iso()})
    write_json(manifest_path, payload)
    return embedding_path, manifest_path
